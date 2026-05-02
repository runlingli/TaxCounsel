"""
Phase 3: Hybrid retrieval — BM25 (sparse) + dense → RRF fusion → CrossEncoder reranking.

Data flow:
    query
      ├── dense embed  ──────────────────┐
      └── TF-IDF sparse encode ─────────┤
                                         ▼
                                   Qdrant RRF fusion (top-20 candidates)
                                         │
                                         ▼
                                   CrossEncoder rerank (top-k final)
"""

import pickle
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from qdrant_client.models import FusionQuery, Prefetch, SparseVector
from sentence_transformers import CrossEncoder, SentenceTransformer

from app.config import settings

# Pre-fusion candidate pool size — larger = better recall, slower reranker
PREFETCH_LIMIT = 20


class HybridRetriever:
    def __init__(self, tfidf_path: str | Path = "data/tfidf.pkl"):
        self.client = QdrantClient(path=settings.qdrant_local_path)

        print("Loading embedding model ...")
        self.embed_model = SentenceTransformer(settings.embed_model)

        print("Loading CrossEncoder reranker ...")
        self.reranker = CrossEncoder(settings.reranker_model)

        print("Loading TF-IDF vectorizer ...")
        with open(tfidf_path, "rb") as f:
            self.tfidf = pickle.load(f)

    def _dense_vec(self, text: str) -> list[float]:
        return self.embed_model.encode(
            [text], normalize_embeddings=True
        )[0].tolist()

    def _sparse_vec(self, text: str) -> SparseVector:
        mat = self.tfidf.transform([text])
        return SparseVector(
            indices=mat.indices.tolist(),
            values=mat.data.tolist(),
        )

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        tax_year: int = 2024,
    ) -> list[tuple]:
        """
        Returns list of (ScoredPoint, reranker_score) sorted best-first.

        Steps:
          1. Encode query into dense + sparse vectors
          2. Qdrant prefetch 20 candidates from each index
          3. RRF fusion merges the two ranked lists into one (built into Qdrant)
          4. CrossEncoder scores each (query, passage) pair
          5. Sort and return top_k
        """
        k = top_k or settings.top_k

        q_dense  = self._dense_vec(query)
        q_sparse = self._sparse_vec(query)

        tax_filter = Filter(
            must=[FieldCondition(key="tax_year", match=MatchValue(value=tax_year))]
        )

        # Qdrant's built-in RRF fusion — no manual implementation needed
        results = self.client.query_points(
            collection_name=settings.qdrant_collection,
            prefetch=[
                Prefetch(query=q_dense,  using="dense",  limit=PREFETCH_LIMIT),
                Prefetch(query=q_sparse, using="sparse", limit=PREFETCH_LIMIT),
            ],
            query=FusionQuery(fusion="rrf"),
            limit=PREFETCH_LIMIT,
            query_filter=tax_filter,
        )

        if not results.points:
            return []

        # CrossEncoder scores using the child text (same grain as the query vectors).
        # We keep child text for scoring precision; the caller uses parent_text for
        # generation context — see prompts/templates.py build_critic_user / build_answer_user.
        pairs  = [(query, p.payload["text"]) for p in results.points]
        scores = self.reranker.predict(pairs)

        ranked = sorted(
            zip(results.points, scores.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )

        return ranked[:k]

    @staticmethod
    def parent_text(point) -> str:
        """
        Return the parent chunk text if available, otherwise fall back to child text.
        Use this instead of point.payload['text'] when building Critic / LLM context.
        """
        return point.payload.get("parent_text") or point.payload["text"]
