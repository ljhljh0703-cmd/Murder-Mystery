"""
indexer.py — builds and persists a FAISS index over clue texts.

Index layout
------------
- One vector per clue document.
- Metadata (clue id, case id, hop, tags, full text) stored in a parallel JSON sidecar.
- The index file and sidecar are written to  <project_root>/index/
"""

import json
import pickle
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_DIR = Path(__file__).parent.parent / "index"
INDEX_FILE = INDEX_DIR / "clues.faiss"
META_FILE = INDEX_DIR / "clues_meta.pkl"

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _get_model() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL_NAME)


def build_index(clues: list[dict[str, Any]]) -> None:
    """Embed all clue texts and write FAISS index + metadata to disk."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    model = _get_model()
    texts = [c["text"] for c in clues]

    print(f"[indexer] Embedding {len(texts)} clues …")
    embeddings: np.ndarray = model.encode(
        texts, show_progress_bar=True, normalize_embeddings=True
    )

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(
        dim
    )  # inner-product == cosine (vectors are L2-normalised)
    index.add(embeddings.astype(np.float32))

    faiss.write_index(index, str(INDEX_FILE))
    with open(META_FILE, "wb") as f:
        pickle.dump(clues, f)

    print(f"[indexer] Saved index ({index.ntotal} vectors, dim={dim}) → {INDEX_FILE}")


def load_index() -> tuple[faiss.Index, list[dict[str, Any]]]:
    """Load the pre-built FAISS index and metadata from disk."""
    if not INDEX_FILE.exists() or not META_FILE.exists():
        raise FileNotFoundError("Index not found. Run  python build_index.py  first.")
    index = faiss.read_index(str(INDEX_FILE))
    with open(META_FILE, "rb") as f:
        meta: list[dict[str, Any]] = pickle.load(f)
    return index, meta
