"""
Phase 7: RAG evaluation — faithfulness, answer_relevancy, context_recall.

Instead of relying on RAGAS's brittle dependency chain, we implement the
three core metrics directly using our LLM + local embeddings.

  faithfulness:       LLM checks if each claim in the answer is grounded in context
  answer_relevancy:   cosine sim between question embedding and answer embedding
  context_recall:     LLM checks if each ground-truth sentence is covered by context

This mirrors exactly what RAGAS does internally, with full control over the prompts.
Usage (from backend/):
    python scripts/evaluate.py
"""

import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent))


def _check_qdrant():
    try:
        urllib.request.urlopen("http://localhost:6333/healthz", timeout=3)
    except Exception:
        print("ERROR: Qdrant is not running.  docker start qdrant")
        sys.exit(1)


from app.agent.critic import ask as agent_ask
from app.agent.critic import get_retriever
from app.config import settings
from data.eval.ground_truth import EVAL_SAMPLES

TAX_YEAR = 2025


# ---------------------------------------------------------------------------
# Metric implementations
# ---------------------------------------------------------------------------

def _llm(client: OpenAI, system: str, user: str) -> str:
    resp = client.chat.completions.create(
        model=settings.llm_model,
        messages=[{"role": "system", "content": system},
                  {"role": "user",   "content": user}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return resp.choices[0].message.content


def faithfulness_score(client: OpenAI, answer: str, contexts: list[str]) -> float:
    """
    Decompose the answer into atomic claims, then verify each against context.
    Score = (# claims supported) / (# total claims)
    """
    context_str = "\n---\n".join(contexts[:3])   # top-3 chunks for speed

    # Step 1: extract claims
    claims_raw = _llm(client,
        'Extract the factual claims from the answer. Return JSON: {"claims": ["claim1", ...]}',
        f"Answer: {answer}"
    )
    try:
        claims: list[str] = json.loads(claims_raw).get("claims", [])
    except Exception:
        return 0.0
    if not claims:
        return 1.0   # refusal/empty answer has no false claims

    # Step 2: verify each claim against context
    verdicts_raw = _llm(client,
        'For each claim, decide if it is supported by the context. Return JSON: {"verdicts": [true, false, ...]} one per claim.',
        f"Context:\n{context_str}\n\nClaims:\n" +
        "\n".join(f"{i+1}. {c}" for i, c in enumerate(claims))
    )
    try:
        verdicts: list[bool] = json.loads(verdicts_raw).get("verdicts", [])
    except Exception:
        return 0.0

    supported = sum(1 for v in verdicts if v is True or v == "true")
    return supported / len(claims) if claims else 0.0


def context_recall_score(client: OpenAI, ground_truth: str, contexts: list[str]) -> float:
    """
    Split ground truth into sentences, check if each is covered by context.
    Score = (# sentences covered) / (# total sentences)
    """
    context_str = "\n---\n".join(contexts[:3])
    sentences = [s.strip() for s in ground_truth.split(".") if s.strip()]
    if not sentences:
        return 0.0

    covered_raw = _llm(client,
        'For each sentence, decide if its information is present in the context. Return JSON: {"covered": [true, false, ...]} one per sentence.',
        f"Context:\n{context_str}\n\nSentences:\n" +
        "\n".join(f"{i+1}. {s}" for i, s in enumerate(sentences))
    )
    try:
        covered: list[bool] = json.loads(covered_raw).get("covered", [])
    except Exception:
        return 0.0

    n_covered = sum(1 for c in covered if c is True or c == "true")
    return n_covered / len(sentences)


def answer_relevancy_score(embed_model, question: str, answer: str) -> float:
    """Cosine similarity between question and answer embeddings."""
    vecs = embed_model.encode([question, answer], normalize_embeddings=True)
    return float(np.dot(vecs[0], vecs[1]))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    _check_qdrant()

    from sentence_transformers import SentenceTransformer
    print("=== TaxCounsel Evaluation ===\n")
    print("Loading models...")
    embed_model = SentenceTransformer(settings.embed_model)
    llm_client  = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
    retriever   = get_retriever()

    faith_scores, recall_scores, relev_scores = [], [], []

    for i, sample in enumerate(EVAL_SAMPLES, 1):
        q  = sample["question"]
        gt = sample["ground_truth"]
        print(f"[{i}/{len(EVAL_SAMPLES)}] {q[:60]}...")

        response = agent_ask(q, tax_year=TAX_YEAR)
        ans      = response.answer.answer
        raw_docs = retriever.retrieve(response.rewritten_query, tax_year=TAX_YEAR)
        ctx      = [d.payload["text"] for d, _ in raw_docs]

        f = faithfulness_score(llm_client, ans, ctx)
        r = context_recall_score(llm_client, gt, ctx)
        v = answer_relevancy_score(embed_model, q, ans)

        faith_scores.append(f)
        recall_scores.append(r)
        relev_scores.append(v)
        print(f"         faithfulness={f:.2f}  recall={r:.2f}  relevancy={v:.2f}")

    scores = {
        "faithfulness":     round(float(np.mean(faith_scores)),  4),
        "answer_relevancy": round(float(np.mean(relev_scores)),  4),
        "context_recall":   round(float(np.mean(recall_scores)), 4),
    }

    print("\n" + "=" * 44)
    print("RESULTS  (n={})".format(len(EVAL_SAMPLES)))
    print("=" * 44)
    for metric, score in scores.items():
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"  {metric:<20} {bar}  {score:.4f}")
    print("=" * 44)

    out = Path("data/eval/results.json")
    out.write_text(json.dumps({"scores": scores, "n_samples": len(EVAL_SAMPLES)}, indent=2))
    print(f"\nSaved → {out}")
    print("\nResume bullet:")
    print(
        f"  Validated end-to-end quality with custom RAGAS-style evaluation "
        f"(faithfulness {scores['faithfulness']}, "
        f"answer_relevancy {scores['answer_relevancy']}, "
        f"context_recall {scores['context_recall']})"
    )


if __name__ == "__main__":
    main()
