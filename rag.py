import os
import json
from typing import List, Dict, Any, Tuple

import numpy as np
import faiss
import requests
from sentence_transformers import SentenceTransformer

INDEX_DIR = "index"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

TOP_K = 4
MAX_CONTEXT_CHARS = 3500  # keep prompt small for speed


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms


class RAGBot:
    def __init__(self):
        self.model = SentenceTransformer(EMBED_MODEL_NAME)
        self.index = faiss.read_index(os.path.join(INDEX_DIR, "faiss.index"))
        with open(os.path.join(INDEX_DIR, "corpus.json"), "r", encoding="utf-8") as f:
            self.corpus = json.load(f)

    def retrieve(self, question: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
        q_emb = self.model.encode([question], convert_to_numpy=True).astype("float32")
        q_emb = l2_normalize(q_emb)

        scores, idxs = self.index.search(q_emb, top_k)
        results = []
        for score, idx in zip(scores[0].tolist(), idxs[0].tolist()):
            if idx == -1:
                continue
            item = dict(self.corpus[idx])
            item["score"] = float(score)
            results.append(item)
        return results

    def build_context(self, retrieved: List[Dict[str, Any]]) -> str:
        blocks = []
        total = 0
        for r in retrieved:
            src = os.path.basename(r["source"])
            text = r["text"].strip().replace("\n", " ")
            block = f"[SOURCE: {src} | chunk {r['chunk_id']} | score {r['score']:.3f}] {text}"
            if total + len(block) > MAX_CONTEXT_CHARS:
                break
            blocks.append(block)
            total += len(block)
        return "\n".join(blocks)

    def generate(self, question: str, context: str) -> str:
        system = (
            "You are a UC Irvine tour guide assistant.\n"
            "You must answer using ONLY the provided context.\n"
            "If the answer is not in the context, say: "
            "\"I don't know based on the provided tour info.\""
        )

        user = (
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION:\n{question}\n\n"
            "Answer clearly and concisely. If useful, give 1-2 quick tips."
        )

        payload = {
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            "stream": False,
            "options": {
                "temperature": 0.2
            }
        }

        resp = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data["message"]["content"].strip()

    def answer(self, question: str) -> Tuple[str, List[Dict[str, Any]]]:
        retrieved = self.retrieve(question)
        context = self.build_context(retrieved)
        answer = self.generate(question, context)
        return answer, retrieved