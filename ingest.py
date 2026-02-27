import os
import glob
import json
from typing import List, Dict, Any, Tuple

import numpy as np
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

DATA_DIR = "data"
INDEX_DIR = "index"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 900       # characters
CHUNK_OVERLAP = 150    # characters


def read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def read_pdf(path: str) -> str:
    reader = PdfReader(path)
    parts = []
    for page in reader.pages:
        text = page.extract_text() or ""
        parts.append(text)
    return "\n".join(parts)


def load_documents(data_dir: str) -> List[Dict[str, Any]]:
    docs = []
    paths = []
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
        start = max(0, end - overlap)
    return chunks


def build_corpus(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms


def main():
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
    for i in tqdm(range(0, len(texts), 64)):
        batch = texts[i:i+64]
        emb = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        embeddings.append(emb)
    embeddings = np.vstack(embeddings).astype("float32")

    # Cosine similarity via inner product after L2 normalization
    embeddings = l2_normalize(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, os.path.join(INDEX_DIR, "faiss.index"))
    with open(os.path.join(INDEX_DIR, "corpus.json"), "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)

    print("Ingestion complete.")
    print(f"Saved index to {INDEX_DIR}/faiss.index")
    print(f"Saved metadata to {INDEX_DIR}/corpus.json")


if __name__ == "__main__":
    main()