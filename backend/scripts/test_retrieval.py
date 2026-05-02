"""
Smoke-test for Phase 3 retrieval.
Usage (from backend/):
    python scripts/test_retrieval.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.retrieval.retriever import HybridRetriever

retriever = HybridRetriever()

QUERIES = [
    "What is the standard deduction for 2024?",
    "Can I deduct my home office expenses?",
    "Section 280A",                            # exact IRS term — BM25 should shine here
]

for query in QUERIES:
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    results = retriever.retrieve(query, top_k=3)
    for rank, (doc, score) in enumerate(results, 1):
        print(
            f"  [{rank}] score={score:.3f}  "
            f"{doc.payload.get('source','')} p.{doc.payload.get('page','?')} "
            f"| {doc.payload.get('section','')[:50]}"
        )
        print(f"       {doc.payload['text'][:120].strip()} ...")
