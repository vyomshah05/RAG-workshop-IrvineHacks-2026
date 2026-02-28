import os
import glob
import json
from typing import List, Dict, Any, Tuple

import numpy as np
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

# Directory containing source documents (our knowledge base)
DATA_DIR = "data"

# Directory where we will save the FAISS index + metadata
INDEX_DIR = "index"

# Embedding model used to convert text -> vectors
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Chunking configuration
# We split long documents into smaller pieces so retrieval works better.
CHUNK_SIZE = 900       # characters per chunk
CHUNK_OVERLAP = 150    # overlap to avoid cutting context mid-thought


def read_txt(path: str) -> str:
    """
    Reads a .txt or .md file from disk and returns its contents as a string.

    Why this exists:
    - RAG systems operate on raw text.
    - We need a clean way to load text documents before chunking.

    Implementation details:
    - Uses UTF-8 encoding.
    - errors="ignore" prevents crashes if the file contains weird characters.
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def read_pdf(path: str) -> str:
    """
    Extracts text from a PDF file and returns it as a single string.

    Why this exists:
    - Many real-world knowledge bases include PDFs.
    - RAG pipelines must convert PDFs into plain text before embedding.

    How it works:
    - Uses PdfReader to iterate through pages.
    - Extracts text from each page.
    - Joins all pages together into one large string.
    - If a page has no extractable text, we safely fallback to "".
    """
    reader = PdfReader(path)
    parts = []
    for page in reader.pages:
        text = page.extract_text() or ""
        parts.append(text)
    return "\n".join(parts)


def load_documents(data_dir: str) -> List[Dict[str, Any]]:
    """
    Loads all supported documents from the data directory.

    Supported file types:
    - .txt
    - .md
    - .pdf

    Why this exists:
    - This is the entry point of the ingestion pipeline.
    - It collects all knowledge base documents before chunking.

    What it returns:
    A list of dictionaries:
        {
            "path": file_path,
            "text": file_contents
        }

    Important details:
    - Uses glob with recursive=True to search subdirectories.
    - Strips whitespace.
    - Skips empty documents.
    """

    docs = []
    paths = []

    # Recursively find all supported file types
    paths += glob.glob(os.path.join(data_dir, "**/*.txt"), recursive=True)
    paths += glob.glob(os.path.join(data_dir, "**/*.md"), recursive=True)
    paths += glob.glob(os.path.join(data_dir, "**/*.pdf"), recursive=True)

    for p in sorted(paths):
        ext = os.path.splitext(p)[1].lower()

        if ext in [".txt", ".md"]:
            text = read_txt(p)
        elif ext == ".pdf":
            text = read_pdf(p)
        else:
            continue

        text = text.strip()

        if text:
            docs.append({"path": p, "text": text})

    return docs


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Splits long text into smaller overlapping chunks.

    Why chunking is critical:
    - Embedding entire documents reduces retrieval accuracy.
    - Smaller chunks allow fine-grained semantic search.
    - Overlap prevents cutting sentences mid-thought.

    How it works:
    - Sliding window over characters.
    - Each chunk is up to chunk_size characters.
    - Next chunk starts slightly before previous ends (overlap).
    - Stops when end of text is reached.

    Returns:
        List of chunk strings.
    """

    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == n:
            break

        # Move start forward but keep overlap
        start = max(0, end - overlap)

    return chunks


def build_corpus(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Converts full documents into a chunk-level corpus.

    Why this exists:
    - Retrieval operates at the chunk level.
    - We must track which chunk came from which document.

    Output format:
        [
            {
                "source": file_path,
                "chunk_id": chunk_index,
                "text": chunk_text
            },
            ...
        ]

    This metadata is later used for:
    - Citations
    - Debugging retrieval
    """

    corpus = []

    for d in docs:
        chunks = chunk_text(d["text"], CHUNK_SIZE, CHUNK_OVERLAP)

        for i, c in enumerate(chunks):
            corpus.append({
                "source": d["path"],
                "chunk_id": i,
                "text": c
            })

    return corpus


def l2_normalize(x: np.ndarray) -> np.ndarray:
    """
    Applies L2 normalization to embeddings.

    Why this matters:
    - Cosine similarity = normalized dot product.
    - FAISS IndexFlatIP uses inner product.
    - If vectors are L2-normalized,
        inner product == cosine similarity.

    What this does:
    - Divides each vector by its length (norm).
    - Ensures all vectors lie on the unit sphere.

    Returns:
        Normalized embedding matrix.
    """

    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms


def main():
    """
    Main ingestion pipeline.

    This performs the full RAG indexing process:

    1. Load documents
    2. Chunk them
    3. Generate embeddings
    4. Normalize embeddings
    5. Build FAISS index
    6. Save index + metadata to disk

    After running this once,
    the chatbot can perform fast semantic search locally.
    """

    # Ensure index directory exists
    os.makedirs(INDEX_DIR, exist_ok=True)

    print(f"Loading documents from: {DATA_DIR}")
    docs = load_documents(DATA_DIR)

    if not docs:
        raise SystemExit("No documents found in data/. Add .txt/.md/.pdf files and try again.")

    corpus = build_corpus(docs)
    print(f"Loaded {len(docs)} documents -> {len(corpus)} chunks")

    print(f"Loading embedding model: {EMBED_MODEL_NAME}")
    model = SentenceTransformer(EMBED_MODEL_NAME)

    texts = [c["text"] for c in corpus]

    print("Embedding chunks...")
    embeddings = []

    # Batch embedding for efficiency
    for i in tqdm(range(0, len(texts), 64)):
        batch = texts[i:i+64]
        emb = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        embeddings.append(emb)

    embeddings = np.vstack(embeddings).astype("float32")

    # Convert to cosine similarity space
    embeddings = l2_normalize(embeddings)

    # Create FAISS index using inner product
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)

    # Add vectors to index
    index.add(embeddings)

    # Save index to disk
    faiss.write_index(index, os.path.join(INDEX_DIR, "faiss.index"))

    # Save corpus metadata (needed to map index -> text)
    with open(os.path.join(INDEX_DIR, "corpus.json"), "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)

    print("Ingestion complete.")
    print(f"Saved index to {INDEX_DIR}/faiss.index")
    print(f"Saved metadata to {INDEX_DIR}/corpus.json")


if __name__ == "__main__":
    main()