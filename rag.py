import os
import json
from typing import List, Dict, Any, Tuple

import numpy as np
import faiss
import requests
from sentence_transformers import SentenceTransformer

# Where ingest.py saved the FAISS index + the corpus metadata
INDEX_DIR = "index"

# Same embedding model used during ingestion.
# IMPORTANT: retrieval must use the *exact same* embedding model,
# otherwise the query vector will live in a different "vector space"
# and similarity search won't work correctly.
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Ollama runs a local HTTP server. By default it listens on localhost:11434.
# Using env vars makes the demo flexible:
# - you can change model names without editing code
# - you can point to a different host if needed (rare)
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

# Retrieval parameters:
# TOP_K controls how many chunks we fetch from the vector DB.
# More chunks = more context, but slower and potentially noisier.
TOP_K = 4

# Limit context size so the prompt stays small (faster generation + lower memory usage).
# This matters a lot for live demos and small local models.
MAX_CONTEXT_CHARS = 3500  # keep prompt small for speed


def l2_normalize(x: np.ndarray) -> np.ndarray:
    """
    L2-normalizes vectors so that cosine similarity becomes a dot product.

    Why this matters:
    - Our FAISS index uses IndexFlatIP, which computes inner product similarity.
    - If vectors are L2-normalized, inner product == cosine similarity.
    - Cosine similarity is a common metric for comparing embeddings.

    Input:
        x: numpy array shape (N, D) where:
           N = number of vectors
           D = embedding dimension

    Output:
        normalized array where each row has length ~1
    """

    # Compute vector lengths for each row
    # keepdims=True keeps the shape (N, 1) so broadcasting works cleanly.
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12

    # Divide each vector by its norm to put it on the unit sphere
    return x / norms


class RAGBot:
    """
    RAGBot bundles the entire *query-time* RAG pipeline:

    1) Embed the user's question
    2) Retrieve relevant chunks from FAISS
    3) Build a context string (with citations)
    4) Ask the LLM to answer using ONLY that context

    This class is intentionally simple and transparent for workshop teaching.
    """

    def __init__(self):
        """
        Initializes everything needed at runtime.

        What gets loaded:
        - SentenceTransformer embedding model (for query embeddings)
        - FAISS index (fast similarity search over chunk embeddings)
        - corpus.json (maps FAISS index IDs -> original chunk text + metadata)

        Why this is done once:
        - Loading models + indexes is relatively expensive.
        - We do it once when the chatbot starts, then reuse for every question.
        """

        # Load embedding model used for both ingestion and retrieval
        self.model = SentenceTransformer(EMBED_MODEL_NAME)

        # Load FAISS index built by ingest.py (contains chunk embeddings)
        self.index = faiss.read_index(os.path.join(INDEX_DIR, "faiss.index"))

        # Load metadata describing each chunk (same ordering as vectors in FAISS)
        # corpus[i] corresponds to vector i in the FAISS index.
        with open(os.path.join(INDEX_DIR, "corpus.json"), "r", encoding="utf-8") as f:
            self.corpus = json.load(f)

    def retrieve(self, question: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
        """
        Retrieves the most relevant chunks for a given question using semantic search.

        Steps:
        1) Convert question -> embedding vector
        2) Normalize it (so inner product behaves like cosine similarity)
        3) Use FAISS to search for top_k nearest chunk vectors
        4) Convert FAISS results into chunk objects with scores

        Inputs:
            question: user question text
            top_k: number of chunks to retrieve

        Output:
            A list of chunk dictionaries, each containing:
            - source: file path of the original document
            - chunk_id: which chunk it is within that document
            - text: the chunk text
            - score: similarity score between question and chunk
        """

        # Embed the question into the SAME vector space as document chunks
        # model.encode returns shape (1, D) since we pass a list of 1 string.
        q_emb = self.model.encode([question], convert_to_numpy=True).astype("float32")

        # Normalize for cosine similarity
        q_emb = l2_normalize(q_emb)

        # FAISS search:
        # scores shape: (1, top_k)
        # idxs   shape: (1, top_k)
        # idxs are integer positions into corpus (and into the index vectors)
        scores, idxs = self.index.search(q_emb, top_k)

        results = []
        for score, idx in zip(scores[0].tolist(), idxs[0].tolist()):
            # FAISS returns -1 if it couldn't find enough neighbors (rare here)
            if idx == -1:
                continue

            # Copy corpus metadata for that chunk
            item = dict(self.corpus[idx])

            # Add similarity score (higher is more similar for cosine/IP)
            item["score"] = float(score)

            results.append(item)

        return results

    def build_context(self, retrieved: List[Dict[str, Any]]) -> str:
        """
        Builds a single text block ("context") to feed into the LLM prompt.

        Why this exists:
        - LLMs can't directly query FAISS.
        - We must paste the retrieved knowledge into the prompt.
        - We include source + chunk info so we can cite where facts came from.

        What it does:
        - Formats each chunk with a citation header:
          [SOURCE: filename | chunk id | score]
        - Flattens newlines so the prompt is compact
        - Enforces a maximum context size for speed

        Input:
            retrieved: list of chunk dicts from retrieve()

        Output:
            A newline-joined string containing formatted chunks
        """

        blocks = []
        total = 0

        for r in retrieved:
            # Show only the filename (not full path) to keep citations clean
            src = os.path.basename(r["source"])

            # Make the chunk one-line to reduce prompt size and avoid weird formatting
            text = r["text"].strip().replace("\n", " ")

            # Include chunk score for debugging/teaching (optional but great for workshops)
            block = f"[SOURCE: {src} | chunk {r['chunk_id']} | score {r['score']:.3f}] {text}"

            # Hard stop if adding this block would exceed the context budget
            if total + len(block) > MAX_CONTEXT_CHARS:
                break

            blocks.append(block)
            total += len(block)

        return "\n".join(blocks)

    def generate(self, question: str, context: str) -> str:
        """
        Calls the LLM (via Ollama) to generate an answer grounded in retrieved context.

        Core RAG principle:
        - The model is instructed to ONLY use the provided context.
        - If context does not contain the answer, it must say it doesn't know.

        Why this matters:
        - This is how we reduce hallucinations.
        - It makes answers traceable to the retrieved sources.

        Inputs:
            question: user question
            context: formatted retrieved chunks string from build_context()

        Output:
            final answer string produced by the LLM
        """

        # SYSTEM message sets behavior rules for the LLM.
        # This is where we "force" grounding.
        system = (
            "You are a UC Irvine tour guide assistant.\n"
            "You must answer using ONLY the provided context.\n"
            "If the answer is not in the context, say: "
            "\"I don't know based on the provided tour info.\""
        )

        # USER message includes the retrieved context and the actual question.
        user = (
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION:\n{question}\n\n"
            "Answer clearly and concisely. If useful, give 1-2 quick tips."
        )

        # Ollama /api/chat uses a chat-style JSON format.
        # stream=False means we want the whole response in one JSON blob (simpler for demos).
        payload = {
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            "stream": False,
            "options": {
                # Lower temperature = more deterministic + less hallucination-prone
                "temperature": 0.2
            }
        }

        # Send request to Ollama server
        # timeout is generous in case a slower laptop needs a moment.
        resp = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=120)

        # Raise an error if Ollama returns non-200 (helps debugging)
        resp.raise_for_status()

        # Parse JSON response and return the assistant message
        data = resp.json()
        return data["message"]["content"].strip()

    def answer(self, question: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Convenience method that runs the full RAG pipeline for one question.

        Steps:
        1) retrieve(question) -> chunks
        2) build_context(chunks) -> prompt context string
        3) generate(question, context) -> answer text

        Output:
            (answer_text, retrieved_chunks)

        Why return retrieved_chunks too?
        - So the UI (app.py) can print citations/sources.
        - Great for teaching: people can see what the retriever found.
        """

        retrieved = self.retrieve(question)
        context = self.build_context(retrieved)
        answer = self.generate(question, context)
        return answer, retrieved