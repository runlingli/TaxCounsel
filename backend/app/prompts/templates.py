"""
Prompt Engineering Layer — all LLM prompts live here.

Why centralize prompts?
- Version-controllable: changing a prompt = a diff you can review
- Testable in isolation: import and unit-test prompt templates
- Swappable: swap GPT-4o-mini for Claude by changing one line, not hunting strings
"""

# ---------------------------------------------------------------------------
# 1. Query Rewriter
#    Goal: translate colloquial user questions into precise IRS terminology
#    so BM25 sparse retrieval can match IRS-specific keywords.
# ---------------------------------------------------------------------------

QUERY_REWRITER_SYSTEM = """\
You are a US tax terminology expert. Rewrite the user's question using \
precise IRS terminology so it matches language found in IRS publications. \
Return ONLY the rewritten question — no explanations.\
"""

QUERY_REWRITER_RETRY_TEMPLATE = """\
{question}

Previous retrieval was insufficient because: {feedback}
Rewrite the question using DIFFERENT IRS terminology or a more specific angle.\
"""


# ---------------------------------------------------------------------------
# 2. LLM Critic
#    Goal: replace a hard confidence threshold with model-based sufficiency
#    evaluation. Returns JSON so we can parse the reason for the retry loop.
# ---------------------------------------------------------------------------

CRITIC_SYSTEM = """\
You are a quality evaluator for a US tax advisory RAG system.
Assess whether the retrieved IRS excerpts contain enough information to \
give a useful, accurate answer to the user's question.

Rules:
- Mark sufficient=true if the excerpts contain the key facts needed, \
  even if some details are missing or the text is fragmented.
- Mark sufficient=false only if the excerpts are entirely off-topic \
  or contain zero relevant information.
- Partial information that allows a directionally correct answer counts as sufficient.

Respond with JSON only — no markdown, no extra text:
{"sufficient": true | false, "reason": "<brief explanation>"}\
"""

CRITIC_USER_TEMPLATE = """\
Question: {question}

Retrieved excerpts:
{excerpts}\
"""


# ---------------------------------------------------------------------------
# 3. Answer Generator
#    Goal: synthesize a grounded, cited answer from IRS context.
#    The prompt enforces citation format and scope constraints.
# ---------------------------------------------------------------------------

ANSWER_SYSTEM = """\
You are a professional US tax advisor. Follow these rules:
1. Answer ONLY from the provided IRS context — never from general knowledge.
2. Cite every claim as: 'IRS Pub XX · Section/Chapter · p.N'.
3. If the answer is not in the context, say so explicitly.
4. Write clearly for a general audience; avoid excessive jargon.\
"""

ANSWER_USER_TEMPLATE = """\
IRS Context:
{context}

Question: {question}\
"""


# ---------------------------------------------------------------------------
# Helper: build critic user message
# ---------------------------------------------------------------------------

def build_critic_user(question: str, docs: list) -> str:
    # Use parent_text so the Critic sees 800 words of context per result,
    # not just the 200-word child chunk that was retrieved.
    excerpts = "\n".join([
        f"[{i+1}] (source={d.payload.get('source','?')} p.{d.payload.get('parent_page', d.payload.get('page','?'))}) "
        f"{(d.payload.get('parent_text') or d.payload['text'])[:1500]}"
        for i, (d, _) in enumerate(docs)
    ])
    return CRITIC_USER_TEMPLATE.format(question=question, excerpts=excerpts)


def build_answer_user(question: str, docs: list) -> str:
    # Deduplicate by parent_text so we don't send the same parent twice
    # when multiple children from the same parent are retrieved.
    seen: set[str] = set()
    unique_docs = []
    for d, s in docs:
        key = d.payload.get("parent_text", d.payload["text"])[:100]
        if key not in seen:
            seen.add(key)
            unique_docs.append((d, s))

    context = "\n\n".join([
        f"[{d.payload.get('source','?')} · "
        f"{d.payload.get('parent_section', d.payload.get('section',''))} · "
        f"p.{d.payload.get('parent_page', d.payload.get('page','?'))}]\n"
        f"{d.payload.get('parent_text') or d.payload['text']}"
        for d, _ in unique_docs
    ])
    return ANSWER_USER_TEMPLATE.format(context=context, question=question)
