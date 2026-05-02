"""
LLM Critic Agent — LangGraph state machine with:
  - Multi-turn conversation history
  - LRU answer cache
  - GraphRAG cross-reference expansion
  - Streaming-compatible step callbacks

State machine:
  START → rewrite_query → retrieve → [graphrag_expand] → critic_evaluate
            ↑ (retry with feedback)        ↓ sufficient
          rewrite_query             generate_answer → END
                  ↓ exhausted
           structured_refusal → END
"""

import json
import hashlib
from functools import lru_cache
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from openai import OpenAI

from app.config import settings
from app.models import AskResponse, RetrievedDoc, TaxAnswer
from app.prompts.templates import (
    ANSWER_SYSTEM,
    CRITIC_SYSTEM,
    QUERY_REWRITER_RETRY_TEMPLATE,
    QUERY_REWRITER_SYSTEM,
    build_answer_user,
    build_critic_user,
)
from app.retrieval.retriever import HybridRetriever

_retriever: HybridRetriever | None = None


def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever


def _llm() -> OpenAI:
    return OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)


# ---------------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    original_question: str
    tax_year: int
    chat_history: list[dict]   # [{"role": "user"|"assistant", "content": str}]
    current_query: str
    retrieved_docs: list
    critic_sufficient: bool
    critic_reason: str
    attempts: int
    final_answer: TaxAnswer | None
    stream_cb: object | None   # optional callback(event, data) for SSE


def _emit(state: AgentState, event: str, data: dict) -> None:
    """Fire a streaming callback if one was registered."""
    if state.get("stream_cb"):
        state["stream_cb"](event, data)


# ---------------------------------------------------------------------------
# Node 1: Query Rewriter
# ---------------------------------------------------------------------------

def rewrite_query(state: AgentState) -> AgentState:
    client = _llm()

    # Build history context for the rewriter so it resolves pronouns / ellipsis
    history_ctx = ""
    if state["chat_history"]:
        history_ctx = "Previous conversation:\n" + "\n".join(
            f"{m['role'].capitalize()}: {m['content']}"
            for m in state["chat_history"][-4:]   # last 2 turns
        ) + "\n\n"

    if state["attempts"] == 0:
        user_content = history_ctx + state["original_question"]
    else:
        user_content = (
            history_ctx
            + QUERY_REWRITER_RETRY_TEMPLATE.format(
                question=state["original_question"],
                feedback=state["critic_reason"],
            )
        )

    resp = client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": QUERY_REWRITER_SYSTEM},
            {"role": "user",   "content": user_content},
        ],
        temperature=0,
    )
    rewritten = resp.choices[0].message.content.strip()
    print(f"  [rewrite #{state['attempts']+1}] {rewritten}")
    _emit(state, "rewriting", {"query": rewritten, "attempt": state["attempts"] + 1})

    return {**state, "current_query": rewritten}


# ---------------------------------------------------------------------------
# Node 2: Hybrid Retriever + GraphRAG cross-reference expansion
# ---------------------------------------------------------------------------

def retrieve(state: AgentState) -> AgentState:
    retriever = get_retriever()
    _emit(state, "retrieving", {"query": state["current_query"]})
    docs = retriever.retrieve(state["current_query"], tax_year=state["tax_year"])
    return {**state, "retrieved_docs": docs}


def graphrag_expand(state: AgentState) -> AgentState:
    """
    Light GraphRAG: if any retrieved chunk references another IRS publication
    (e.g. 'See Publication 596'), fetch a top result from that publication too.

    This stitches cross-publication knowledge without a full graph database.
    """
    retriever = get_retriever()
    docs = list(state["retrieved_docs"])
    existing_ids = {d.id for d, _ in docs}

    # Collect unique cross-references from retrieved chunks
    xrefs: list[str] = []
    for doc, _ in docs:
        xrefs.extend(doc.payload.get("cross_refs", []))

    added = 0
    for xref in dict.fromkeys(xrefs):   # deduplicate, preserve order
        if added >= 2:                   # cap expansion at 2 extra docs
            break
        xref_query = f"{state['current_query']} {xref}"
        xref_docs = retriever.retrieve(xref_query, top_k=1, tax_year=state["tax_year"])
        for d, s in xref_docs:
            if d.id not in existing_ids:
                docs.append((d, s))
                existing_ids.add(d.id)
                added += 1
                print(f"  [graphrag] expanded via ref '{xref}' → {d.payload.get('source','')} p.{d.payload.get('page','?')}")

    return {**state, "retrieved_docs": docs}


# ---------------------------------------------------------------------------
# Node 3: LLM Critic
# ---------------------------------------------------------------------------

def critic_evaluate(state: AgentState) -> AgentState:
    client = _llm()
    _emit(state, "evaluating", {})

    resp = client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": CRITIC_SYSTEM},
            {"role": "user",   "content": build_critic_user(
                state["original_question"], state["retrieved_docs"]
            )},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    result = json.loads(resp.choices[0].message.content)
    sufficient: bool = result.get("sufficient", False)
    reason: str      = result.get("reason", "")
    print(f"  [critic] sufficient={sufficient}  reason={reason!r}")

    return {
        **state,
        "critic_sufficient": sufficient,
        "critic_reason": reason,
        "attempts": state["attempts"] + 1,
    }


# ---------------------------------------------------------------------------
# Node 4a: Answer Generator
# ---------------------------------------------------------------------------

def generate_answer(state: AgentState) -> AgentState:
    client = _llm()
    docs = state["retrieved_docs"]
    _emit(state, "generating", {})

    citations = list({
        f"{d.payload.get('source','?')} · "
        f"{d.payload.get('parent_section', d.payload.get('section',''))} · "
        f"p.{d.payload.get('parent_page', d.payload.get('page','?'))}"
        for d, _ in docs
    })

    schema_hint = (
        'Respond with JSON: {"answer": string, "page_citations": [string], '
        '"tax_year": int, "confidence": float, "disclaimer": string}. No markdown.'
    )

    resp = client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": ANSWER_SYSTEM + "\n\n" + schema_hint},
            {"role": "user",   "content": build_answer_user(
                state["original_question"], docs
            )},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    raw = json.loads(resp.choices[0].message.content)
    answer = TaxAnswer(
        answer=        raw.get("answer", ""),
        page_citations=citations,
        tax_year=      state["tax_year"],
        confidence=    docs[0][1] if docs else 0.0,
        disclaimer=    raw.get("disclaimer", TaxAnswer.model_fields["disclaimer"].default),
    )
    return {**state, "final_answer": answer}


# ---------------------------------------------------------------------------
# Node 4b: Structured Refusal
# ---------------------------------------------------------------------------

def structured_refusal(state: AgentState) -> AgentState:
    return {**state, "final_answer": TaxAnswer(
        answer=(
            "This question falls outside the current IRS knowledge base. "
            "Please consult IRS.gov directly or a licensed tax professional."
        ),
        page_citations=[], tax_year=state["tax_year"], confidence=0.0,
    )}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def route_after_critic(state: AgentState) -> str:
    if state["critic_sufficient"]:
        return "generate_answer"
    if state["attempts"] < settings.max_retries:
        return "rewrite_query"
    return "structured_refusal"


# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    g = StateGraph(AgentState)
    g.add_node("rewrite_query",      rewrite_query)
    g.add_node("retrieve",           retrieve)
    g.add_node("graphrag_expand",    graphrag_expand)
    g.add_node("critic_evaluate",    critic_evaluate)
    g.add_node("generate_answer",    generate_answer)
    g.add_node("structured_refusal", structured_refusal)

    g.add_edge(START,              "rewrite_query")
    g.add_edge("rewrite_query",    "retrieve")
    g.add_edge("retrieve",         "graphrag_expand")
    g.add_edge("graphrag_expand",  "critic_evaluate")
    g.add_conditional_edges("critic_evaluate", route_after_critic)
    g.add_edge("generate_answer",    END)
    g.add_edge("structured_refusal", END)
    return g.compile()


graph = build_graph()


# ---------------------------------------------------------------------------
# LRU cache key helper
# ---------------------------------------------------------------------------

def _cache_key(question: str, tax_year: int) -> str:
    return hashlib.md5(f"{question.lower().strip()}|{tax_year}".encode()).hexdigest()


_answer_cache: dict[str, AskResponse] = {}
CACHE_MAX = 256


def _get_cached(key: str) -> AskResponse | None:
    return _answer_cache.get(key)


def _set_cached(key: str, resp: AskResponse) -> None:
    if len(_answer_cache) >= CACHE_MAX:
        # Evict oldest entry (simple FIFO)
        _answer_cache.pop(next(iter(_answer_cache)))
    _answer_cache[key] = resp


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def ask(
    question: str,
    tax_year: int = 2025,
    chat_history: list[dict] | None = None,
    stream_cb=None,
) -> AskResponse:
    """
    Run the full agent pipeline.

    chat_history: list of {"role": "user"|"assistant", "content": str}
    stream_cb: optional callable(event: str, data: dict) fired at each step
    """
    # Cache lookup only for single-turn queries (history changes meaning)
    cache_key = _cache_key(question, tax_year) if not chat_history else None
    if cache_key:
        cached = _get_cached(cache_key)
        if cached:
            print(f"  [cache HIT] {question[:50]}")
            return cached

    print(f"\n[agent] question={question!r}  tax_year={tax_year}")
    init_state: AgentState = {
        "original_question": question,
        "tax_year":          tax_year,
        "chat_history":      chat_history or [],
        "current_query":     question,
        "retrieved_docs":    [],
        "critic_sufficient": False,
        "critic_reason":     "",
        "attempts":          0,
        "final_answer":      None,
        "stream_cb":         stream_cb,
    }

    final: AgentState = graph.invoke(init_state)

    docs = final["retrieved_docs"]
    response = AskResponse(
        answer=          final["final_answer"],
        original_query=  question,
        rewritten_query= final["current_query"],
        retrieved_docs=  [
            RetrievedDoc(
                source=  d.payload.get("source", ""),
                section= d.payload.get("parent_section", d.payload.get("section", "")),
                page=    int(d.payload.get("parent_page", d.payload.get("page", 0))),
                score=   round(float(s), 4),
                excerpt= d.payload["text"][:300],
            )
            for d, s in docs
        ],
        attempts= final["attempts"],
    )

    if cache_key:
        _set_cached(cache_key, response)

    return response
