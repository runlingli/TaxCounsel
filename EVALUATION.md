# TaxCounsel — System Evaluation Report

**Version:** 0.3.0  
**Evaluation Date:** 2026-05-01  
**Knowledge Base:** 9 IRS Publications (2025 tax year)  
**Answer Rate:** 11 / 12 (91.7%)

---

## 1. System Architecture

TaxCounsel is a production-grade Retrieval-Augmented Generation (RAG) system for US tax advisory, built on a hybrid retrieval pipeline with an LLM critic agent.

```
User Question
     │
     ▼
┌─────────────────────────────────────────────────────┐
│              LangGraph State Machine                │
│                                                     │
│  START → rewrite_query → retrieve → graphrag_expand │
│              ↑ retry            ↓                   │
│          rewrite_query    critic_evaluate            │
│                               ↓ sufficient          │
│                         generate_answer → END       │
│                               ↓ exhausted           │
│                         structured_refusal → END    │
└─────────────────────────────────────────────────────┘
     │
     ▼
SSE Stream (rewriting → retrieving → evaluating → generating → done)
```

### Components

| Layer | Technology | Role |
|---|---|---|
| Frontend | Next.js 16 (App Router) | Multi-turn chat UI, streaming status, feedback |
| API | FastAPI + uvicorn | REST + Server-Sent Events |
| Orchestration | LangGraph | Retry state machine |
| LLM | DeepSeek-chat | Query rewriting, critic evaluation, answer generation |
| Embeddings | BAAI/bge-large-en-v1.5 (1024-dim) | Dense vector retrieval |
| Sparse | TF-IDF (30k vocab) | BM25-style keyword matching |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 | Precision scoring |
| Vector DB | Qdrant (local on-disk) | Hybrid dense + sparse index, no Docker required |
| Embed Cache | NumPy `.npz` checkpoint | Skip re-encoding on restart |
| Feedback | SQLite | Thumbs up/down persistence |
| Cache | In-memory LRU (256 entries) | Repeat query acceleration |

---

## 2. Knowledge Base

### Publications Ingested

| Publication | Topic | Chunks |
|---|---|---|
| IRS Pub 17 (2025) | Your Federal Income Tax — general guide | 1,497 |
| IRS Pub 334 (2025) | Tax Guide for Small Business (Self-Employed) | 403 |
| IRS Pub 463 (2025) | Travel, Gift, and Car Expenses | 429 |
| IRS Pub 501 (2025) | Dependents, Standard Deduction, Filing Info | 306 |
| IRS Pub 502 (2025) | Medical and Dental Expenses | 172 |
| IRS Pub 525 (2025) | Taxable and Nontaxable Income | 482 |
| IRS Pub 531 (2025) | Reporting Tip Income | 69 |
| IRS Pub 550 (2025) | Investment Income and Expenses | 917 |
| IRS Pub 596 (2025) | Earned Income Credit (EIC) | 343 |
| **Total** | | **4,618** |

### Chunking Strategy: Semantic Parent-Child

Each PDF is parsed with a two-level hierarchy:

- **Parent chunks** (~500 words): paragraph-aware splits at IRS section boundaries. Sent to the LLM critic and answer generator for wide context.
- **Child chunks** (150 words, 30-word overlap): precise retrieval units matched against query embeddings and BM25 sparse vectors.

Each child stores `parent_text` in metadata, so the critic sees 500-word context windows while retrieval uses 150-word precision units. Cross-references (e.g. "See Publication 596") are extracted and stored in `cross_refs` metadata for GraphRAG expansion.

---

## 3. Retrieval Pipeline

### Step 1 — Query Rewriting
The LLM rewrites colloquial questions into precise IRS terminology before retrieval:

| User Input | Rewritten Query |
|---|---|
| "Can I deduct medical expenses?" | "Are medical expenses deductible as an itemized deduction on Schedule A (Form 1040)?" |
| "What is self-employment tax?" | "What is the self-employment tax (SECA tax) and how is the net earnings from self-employment computed?" |
| "What about if I'm 66 years old?" | "What is the additional standard deduction for a single filer age 65 or older in 2025?" |
| "What is the capital gains tax rate?" | "What are the tax rates for net capital gain under Internal Revenue Code Section 1(h)?" |

### Step 2 — Hybrid Retrieval (RRF Fusion)
Two retrieval signals are fused via Reciprocal Rank Fusion:

1. **Dense**: BAAI/bge-large-en-v1.5 cosine similarity (1024-dim)
2. **Sparse**: TF-IDF bag-of-words (30k vocab), fitted on the full corpus

RRF normalizes ranks from both signals and returns the top-K fused candidates.

### Step 3 — GraphRAG Cross-Reference Expansion
After retrieval, the agent scans retrieved chunks for IRS cross-references (regex: `See Publication \d+`, `See Chapter \d+`). For each unique reference, it fetches one additional top-1 result from that publication. This stitches multi-publication knowledge without a full graph database, capped at 2 additional docs per query.

### Step 4 — Cross-Encoder Reranking
Retrieved candidates are reranked by `cross-encoder/ms-marco-MiniLM-L-6-v2`, which scores query-document relevance directly (not just vector similarity). The top-5 results proceed to the critic.

### Step 5 — LLM Critic Evaluation
An LLM critic reads the top-5 results (1,500-char parent context per document) and returns:
```json
{"sufficient": true, "reason": "Excerpts contain the 7.5% AGI threshold and Schedule A instructions."}
```
If insufficient, the agent rewrites the query with the critic's feedback and retries (max 2 retries). If exhausted, a structured refusal is returned.

### Step 6 — Answer Generation
The answer generator receives deduplicated parent-text contexts and produces a JSON response:
```json
{
  "answer": "...",
  "page_citations": ["IRS Pub 550 · Part II of Form 8949 · p.96"],
  "tax_year": 2025,
  "confidence": 4.17,
  "disclaimer": "For informational purposes only. Consult a licensed tax professional."
}
```

---

## 4. Prompt Engineering

All prompts are centralized in `backend/app/prompts/templates.py` for version-controllable, testable, and model-swappable design.

### Query Rewriter Prompt
```
You are a US tax terminology expert. Rewrite the user's question using
precise IRS terminology so it matches language found in IRS publications.
Return ONLY the rewritten question — no explanations.
```
**Design rationale:** BM25 sparse retrieval matches exact tokens. Colloquial terms like "deduct" vs. "Schedule A itemized deduction" produce dramatically different keyword matches. The rewriter bridges natural language to IRS vocabulary.

### LLM Critic Prompt
```
You are a quality evaluator for a US tax advisory RAG system.
Assess whether the retrieved IRS excerpts contain enough information to
give a useful, accurate answer to the user's question.

Rules:
- Mark sufficient=true if the excerpts contain the key facts needed,
  even if some details are missing or the text is fragmented.
- Mark sufficient=false only if the excerpts are entirely off-topic
  or contain zero relevant information.
- Partial information that allows a directionally correct answer counts as sufficient.

Respond with JSON only: {"sufficient": true | false, "reason": "<brief explanation>"}
```
**Design rationale:** Replaces a hard confidence threshold with model-based evaluation. The critic's `reason` field feeds into the retry query rewrite, creating a feedback loop that improves retrieval on each attempt.

### Answer Generator Prompt
```
You are a professional US tax advisor. Follow these rules:
1. Answer ONLY from the provided IRS context — never from general knowledge.
2. Cite every claim as: 'IRS Pub XX · Section/Chapter · p.N'.
3. If the answer is not in the context, say so explicitly.
4. Write clearly for a general audience; avoid excessive jargon.
```
**Design rationale:** Strict grounding constraint prevents hallucination. Mandatory citation format makes every claim traceable to an IRS page.

---

## 5. Evaluation Results

### Test Suite (12 questions, 2025 tax year)

| # | Topic | Answered | Conf | Attempts | Time | Rewritten Query |
|---|---|:---:|---:|:---:|---:|---|
| 1 | Standard Deduction (Single) | ✅ | 5.91 | 1 | 9.5s | "What is the standard deduction amount for a single filing status for tax year 2025?" |
| 2 | Standard Deduction (MFJ) | ✅ | 7.42 | 1 | 8.4s | "What is the standard deduction amount for taxpayers filing as married filing jointly for tax year 2025?" |
| 3 | Filing Requirement | ✅ | 5.86 | 1 | 9.6s | "Who must file a federal income tax return?" |
| 4 | Medical Deductions | ✅ | 7.59 | 1 | 7.9s | "Are medical expenses deductible as an itemized deduction on Schedule A (Form 1040)?" |
| 5 | Business Travel | ✅ | 3.30 | 1 | 8.7s | "What are the requirements for deducting ordinary and necessary business travel expenses?" |
| 6 | EITC Qualification | ✅ | 3.61 | 1 | 10.4s | "What are the eligibility requirements to claim the earned income tax credit?" |
| 7 | Child Tax Credit | ✅ | 6.49 | 1 | 8.2s | "What is the Child Tax Credit (CTC)?" |
| 8 | Self-Employment Tax | ✅ | 5.42 | 1 | 9.1s | "What is the self-employment tax (SECA tax) and how is the net earnings from self-employment computed?" |
| 9 | SE Deductions | ❌ | 0.00 | 2 | 10.7s | *Retrieval insufficient — off-topic chunks rank above Pub 334 after corpus expansion* |
| 10 | Multi-turn (age follow-up) | ✅ | 5.59 | 1 | 9.3s | "What is the additional standard deduction for a single filer age 65 or older in 2025?" |
| 11 | Capital Gains Tax Rate | ✅ | 4.17 | 2 | 10.4s | "What are the tax rates for net capital gain under Internal Revenue Code Section 1(h)?" |
| 12 | Tip Income Reporting | ✅ | 5.34 | 1 | 8.5s | "How do I report tip income on my federal income tax return?" |

### Summary Statistics

| Metric | Value |
|---|---|
| **Answer Rate** | **11 / 12 (91.7%)** |
| Avg. Confidence (answered) | 5.52 |
| Avg. Response Time | 9.2s |
| Avg. Retrieval Attempts | 1.2 |
| Single-attempt success | 10 / 11 answered (91%) |
| Cache speedup (repeat query) | ~480× (48.5s → 0.1s) |

### Sample Answers

**Medical Deductions** (conf=7.59):
> "Yes, you can deduct medical and dental expenses as an itemized deduction on Schedule A (Form 1040), but only the amount that exceeds 7.5% of your adjusted gross income (AGI)."
> — *IRS Pub 502 · General · p.1*

**Standard Deduction MFJ** (conf=7.42):
> "For married couples filing jointly in 2025, the standard deduction is $31,500. If one or both spouses are age 65 or older or blind, the deduction increases: for one qualifying spouse, it is $33,100; for both, it is $34,700."
> — *IRS Pub 501 · General · p.24*

**Capital Gains Tax Rate** (conf=4.17):
> "The capital gains tax rate depends on the type of gain and your taxable income. For 2025, the maximum capital gain rates are 0%, 15%, 20%, 25%, and 28%. Specifically: collectibles gain and eligible gain on qualified small business stock minus the section 1202 exclusion are taxed at 28%; unrecaptured section 1250 gain at 25%; and other gain at 0%, 15%, or 20% based on your taxable income and filing status. See Table 4-4 in IRS Pub 550 for the exact income thresholds."
> — *IRS Pub 550 · p.95–96, 103*

**Multi-turn (age follow-up)** (conf=5.59):
> "If you are 66 years old on the last day of 2025, you are eligible for a higher standard deduction because you are age 65 or older. The standard deduction amount depends on your filing status and whether you or your spouse is also blind."
> — *IRS Pub 501 · General · p.23*

### Known Gap

**SE Deductions (Q9)** — Pub 334 covers Schedule C deductions by category across many sections, but broad queries ("all deductible business expenses") pull off-topic chunks to the top after the corpus expanded to 4,618 chunks. Pub 502 p.20 scores highest due to incidental "ordinary and necessary" phrasing in a medical expense recovery context. Fix: add **Pub 535** (Business Expenses), which contains a consolidated Schedule C deduction summary that would rank correctly.

---

## 6. Features

### Backend
- **Hybrid RAG**: Dense (BAAI/bge-large-en-v1.5) + Sparse (TF-IDF) with RRF fusion
- **Parent-Child Chunking**: 150-word retrieval units backed by 500-word generation context
- **LLM Critic Agent**: Model-based sufficiency evaluation with structured JSON output
- **Query Rewriting**: IRS terminology normalization before every retrieval
- **GraphRAG Expansion**: Cross-reference following across publications
- **Multi-turn Conversations**: Chat history fed into query rewriter for anaphora resolution
- **Streaming SSE**: Real-time events (`rewriting` → `retrieving` → `evaluating` → `generating` → `done`)
- **LRU Answer Cache**: 256-entry in-memory cache keyed by `md5(question|tax_year)`
- **Embedding Cache**: NumPy `.npz` checkpoint — re-ingestion skips the ~20-min encoding phase
- **Local Qdrant**: On-disk mode (`data/qdrant_db/`), no Docker required
- **Feedback Collection**: SQLite storage of thumbs up/down with export endpoint
- **Structured Refusal**: Out-of-scope questions return a graceful, honest refusal

### Frontend
- **Multi-turn Chat**: Conversation history with user/assistant bubbles
- **Streaming Status**: Live step labels during generation (no spinner, skeleton loaders)
- **Retrieval Trace**: Collapsible panel with reranker scores, source pages, and query rewrite
- **Confidence Bar**: Color-coded score bar (green ≥ 60%, amber ≥ 30%, red < 30%)
- **Page Citations**: Amber badges with BookOpen icon for every cited IRS page
- **Feedback Buttons**: Per-turn thumbs up/down, disabled after submission
- **Dark/Light Mode**: System-preference toggle
- **Starter Questions**: One-click topic cards on empty state
- **Tax Year Selector**: Toggle between 2023–2025

---

## 7. Retrieval Quality Notes

### Confidence Score Distribution
The cross-encoder reranker score is the primary quality signal, not a probability:
- `> 5.0`: Strong match — direct answer found in top document
- `1.0 – 5.0`: Moderate match — answer synthesized from multiple chunks
- `< 1.0`: Weak match — critic is lenient; answer may be partial
- `0.0`: Refusal — critic found all documents insufficient after max retries

### Query Rewrite Effectiveness
The rewriter successfully resolves:
- **Colloquial phrasing**: "Can I deduct?" → "Are X expenses deductible as an itemized deduction on Schedule A?"
- **Pronoun/ellipsis resolution** (multi-turn): "What about if I'm 66?" → full standalone query with filing status and year
- **Terminology normalization**: "capital gains tax" → "tax rates for net capital gain under Internal Revenue Code Section 1(h)"

---

## 8. Running the System

```bash
# Backend (Qdrant runs locally — no Docker needed)
cd backend
source .venv/bin/activate
uvicorn app.main:app --port 8000

# Frontend
cd frontend
npm run dev

# Re-ingest (after adding new PDFs to data/irs_pdfs/)
cd backend
# If corpus changed, delete cache first:
rm data/embed_cache.npz data/tfidf_cache.pkl
python scripts/ingest.py
# First run: encodes all chunks (~20 min), saves embed_cache.npz
# Subsequent runs with same corpus: loads cache, skips encoding (~2 min)
```

### API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/ask` | Full RAG pipeline, JSON response |
| `POST` | `/ask/stream` | Same pipeline, SSE streaming events |
| `POST` | `/feedback` | Store thumbs up/down |
| `GET` | `/feedback/export` | Export all feedback (JSON) |
| `GET` | `/cache/stats` | LRU cache size |
| `DELETE` | `/cache` | Clear answer cache |
| `GET` | `/health` | Liveness check |
| `GET` | `/collections` | Qdrant collection info |

---

## 9. Extending the Knowledge Base

To add a new IRS publication:

1. Download the PDF:
   ```bash
   curl -o backend/data/irs_pdfs/p535.pdf https://www.irs.gov/pub/irs-pdf/p535.pdf
   ```

2. Add to `backend/scripts/ingest.py`:
   ```python
   ("data/irs_pdfs/p535.pdf", "IRS Pub 535", 2025, False),  # Business expenses
   ```

3. Delete the embedding cache (corpus changed, must re-encode):
   ```bash
   rm backend/data/embed_cache.npz backend/data/tfidf_cache.pkl
   ```

4. Re-run ingestion:
   ```bash
   python scripts/ingest.py
   ```

5. Restart the API server.

The next publication that would most improve coverage: **Pub 535** (Business Expenses — consolidated Schedule C deduction list, would fix the SE Deductions gap in Q9).
