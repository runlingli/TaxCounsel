"""
Phase 2: Build hybrid (dense + sparse) index in Qdrant.

Two vector spaces in one collection:
  - "dense":  1024-dim cosine vectors from BAAI/bge-large-en-v1.5
  - "sparse": TF-IDF bag-of-words vectors (BM25-style keyword matching)

Having both lets Phase 3 run RRF fusion — dense catches semantic meaning,
sparse catches exact IRS keyword matches that dense embeddings often miss.
"""

import pickle
from pathlib import Path

import numpy as np
from llama_index.core.schema import TextNode
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from app.config import settings

SPARSE_VOCAB_SIZE = 30_000
BATCH_SIZE = 64   # number of texts to embed at once (memory vs. speed trade-off)


def _build_client() -> QdrantClient:
    path = Path(settings.qdrant_local_path)
    path.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(path))


def create_collection(client: QdrantClient) -> None:
    """Create a fresh collection that supports both dense and sparse vectors."""
    existing = [c.name for c in client.get_collections().collections]
    if settings.qdrant_collection in existing:
        client.delete_collection(settings.qdrant_collection)

    client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config={
            "dense": VectorParams(size=1024, distance=Distance.COSINE)
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(
                index=SparseIndexParams(on_disk=False)
            )
        },
    )
    print(f"Collection '{settings.qdrant_collection}' created.")


EMBED_CACHE = Path("data/embed_cache.npz")
TFIDF_CACHE = Path("data/tfidf_cache.pkl")


def build_and_upload(
    nodes: list[TextNode],
    tfidf_save_path: str | Path = "data/tfidf.pkl",
) -> TfidfVectorizer:
    """
    Encode all nodes, fit TF-IDF vocab on the corpus, upload to Qdrant.
    Embeddings are checkpointed to disk so Docker restarts don't require re-encoding.
    """
    texts = [n.text for n in nodes]

    # --- Dense embeddings (with disk checkpoint) ---
    if EMBED_CACHE.exists() and TFIDF_CACHE.exists():
        cached = np.load(EMBED_CACHE)
        if cached["dense_vecs"].shape[0] == len(texts):
            print(f"Loading cached embeddings from {EMBED_CACHE} ...")
            dense_vecs = cached["dense_vecs"]
            with open(TFIDF_CACHE, "rb") as f:
                tfidf = pickle.load(f)
            sparse_matrix = tfidf.transform(texts)
        else:
            print("Cache size mismatch — re-encoding ...")
            dense_vecs, tfidf, sparse_matrix = _encode(texts)
            _save_cache(dense_vecs, tfidf)
    else:
        dense_vecs, tfidf, sparse_matrix = _encode(texts)
        _save_cache(dense_vecs, tfidf)

    # Persist the fitted vectorizer for the retriever
    with open(tfidf_save_path, "wb") as f:
        pickle.dump(tfidf, f)
    print(f"TF-IDF vectorizer saved → {tfidf_save_path}")

    # --- Qdrant upload ---
    client = _build_client()
    create_collection(client)

    print("Uploading points to Qdrant ...")
    points: list[PointStruct] = []
    for i, (node, dv) in enumerate(zip(nodes, dense_vecs)):
        sv = sparse_matrix[i]
        points.append(PointStruct(
            id=i,
            vector={
                "dense":  dv.tolist(),
                "sparse": SparseVector(
                    indices=sv.indices.tolist(),
                    values=sv.data.tolist(),
                ),
            },
            payload={"text": node.text, **node.metadata},
        ))

    for start in range(0, len(points), 256):
        client.upsert(
            collection_name=settings.qdrant_collection,
            points=points[start : start + 256],
        )

    print(f"Uploaded {len(points)} points to '{settings.qdrant_collection}'.")
    return tfidf


def _encode(texts: list[str]) -> tuple[np.ndarray, TfidfVectorizer, any]:
    print(f"Encoding {len(texts)} chunks with {settings.embed_model} ...")
    embed_model = SentenceTransformer(settings.embed_model)
    dense_vecs: np.ndarray = embed_model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    print("Fitting TF-IDF vectorizer ...")
    tfidf = TfidfVectorizer(max_features=SPARSE_VOCAB_SIZE)
    sparse_matrix = tfidf.fit_transform(texts)
    return dense_vecs, tfidf, sparse_matrix


def _save_cache(dense_vecs: np.ndarray, tfidf: TfidfVectorizer) -> None:
    EMBED_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(EMBED_CACHE, dense_vecs=dense_vecs)
    with open(TFIDF_CACHE, "wb") as f:
        pickle.dump(tfidf, f)
    print(f"Embedding cache saved → {EMBED_CACHE}")
