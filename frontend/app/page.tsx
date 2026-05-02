"use client";

import { useState, useRef, useEffect } from "react";
import {
  Send, ChevronDown, ChevronUp, BookOpen, ThumbsUp, ThumbsDown, RotateCcw
} from "lucide-react";
import { ThemeToggle } from "@/components/ThemeToggle";
import {
  askStream, submitFeedback,
  AskResponse, ChatMessage, StreamEvent,
} from "@/lib/api";

const YEAR_OPTIONS = [2025, 2024, 2023];

const STARTERS = [
  "What is the standard deduction for single filers?",
  "Can I deduct medical expenses?",
  "What is the earned income credit?",
  "How do I report self-employment income?",
];

type Status = "idle" | "rewriting" | "retrieving" | "evaluating" | "generating";

interface Turn {
  id: string;
  question: string;
  status: Status;
  statusText: string;
  response: AskResponse | null;
  error: string | null;
  traceOpen: boolean;
  feedback: "up" | "down" | null;
}

function confidenceBar(score: number) {
  const pct = Math.min(100, Math.max(0, ((score + 10) / 20) * 100));
  const color = pct > 60 ? "var(--accent)" : pct > 30 ? "var(--amber)" : "#ef4444";
  return { pct, color };
}

const STATUS_LABELS: Record<Status, string> = {
  idle:       "",
  rewriting:  "Rewriting query…",
  retrieving: "Retrieving from IRS publications…",
  evaluating: "Critic evaluating context…",
  generating: "Generating answer…",
};

function SkeletonLine({ w }: { w: string }) {
  return (
    <div style={{
      height: 13, width: w, borderRadius: 6,
      background: "var(--bg-subtle)",
      animation: "pulse 1.4s ease-in-out infinite",
      marginBottom: 8,
    }} />
  );
}

function TurnCard({ turn, taxYear }: { turn: Turn; taxYear: number }) {
  const [traceOpen, setTraceOpen] = useState(turn.traceOpen);
  const [fb, setFb] = useState<"up" | "down" | null>(turn.feedback);
  const isRefusal = turn.response?.answer.confidence === 0;

  async function handleFeedback(val: "up" | "down") {
    if (!turn.response || fb) return;
    setFb(val);
    await submitFeedback(
      turn.question,
      turn.response.answer.answer,
      taxYear,
      val === "up",
    );
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      {/* User bubble */}
      <div style={{ display: "flex", justifyContent: "flex-end" }}>
        <div style={{
          background: "linear-gradient(135deg, var(--accent), var(--accent-2))",
          color: "#fff", borderRadius: "14px 14px 4px 14px",
          padding: "10px 16px", maxWidth: "80%", fontSize: 15, lineHeight: 1.5,
        }}>
          {turn.question}
        </div>
      </div>

      {/* Assistant bubble */}
      <div style={{ maxWidth: "88%" }}>
        {/* Loading state */}
        {!turn.response && !turn.error && (
          <div style={{
            background: "var(--bg-card)", border: "1px solid var(--border)",
            borderRadius: "4px 14px 14px 14px", padding: 20,
          }}>
            <div style={{
              fontSize: 12, fontFamily: "monospace", color: "var(--accent)",
              marginBottom: 14, display: "flex", alignItems: "center", gap: 6,
            }}>
              <span style={{ animation: "spin 1s linear infinite", display: "inline-block" }}>⟳</span>
              {STATUS_LABELS[turn.status] || "Thinking…"}
            </div>
            <SkeletonLine w="92%" />
            <SkeletonLine w="78%" />
            <SkeletonLine w="55%" />
            <style>{`
              @keyframes pulse{0%,100%{opacity:.35}50%{opacity:.85}}
              @keyframes spin{from{transform:rotate(0deg)}to{transform:rotate(360deg)}}
            `}</style>
          </div>
        )}

        {/* Error */}
        {turn.error && (
          <div style={{
            background: "#fef2f2", border: "1px solid #fecaca",
            borderRadius: "4px 14px 14px 14px", padding: 16,
            color: "#dc2626", fontSize: 14,
          }}>
            {turn.error}
          </div>
        )}

        {/* Answer */}
        {turn.response && (
          <>
            <div style={{
              background: "var(--bg-card)", border: "1px solid var(--border)",
              borderRadius: "4px 14px 14px 14px", padding: 20,
              borderLeft: isRefusal ? "3px solid var(--amber)" : "3px solid var(--accent)",
            }}>
              <p style={{ fontSize: 15, lineHeight: 1.75, color: "var(--text)", marginBottom: 16 }}>
                {turn.response.answer.answer}
              </p>

              {!isRefusal && (() => {
                const { pct, color } = confidenceBar(turn.response!.answer.confidence);
                return (
                  <div style={{ marginBottom: 14 }}>
                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                      <span style={{ fontSize: 11, color: "var(--text-muted)" }}>Confidence</span>
                      <span style={{ fontSize: 11, fontFamily: "monospace", color }}>{turn.response!.answer.confidence.toFixed(3)}</span>
                    </div>
                    <div style={{ height: 3, background: "var(--bg-subtle)", borderRadius: 2 }}>
                      <div style={{ height: "100%", width: `${pct}%`, background: color, borderRadius: 2, transition: "width 0.5s" }} />
                    </div>
                  </div>
                );
              })()}

              {turn.response.answer.page_citations.length > 0 && (
                <div style={{ display: "flex", flexDirection: "column", gap: 5, marginBottom: 14 }}>
                  {turn.response.answer.page_citations.map((c, i) => (
                    <div key={i} style={{ display: "flex", gap: 6, alignItems: "center" }}>
                      <BookOpen size={11} style={{ color: "var(--amber)", flexShrink: 0 }} />
                      <span style={{ fontSize: 11, fontFamily: "monospace", color: "var(--amber)" }}>{c}</span>
                    </div>
                  ))}
                </div>
              )}

              {/* Feedback */}
              <div style={{
                display: "flex", alignItems: "center", justifyContent: "space-between",
                paddingTop: 12, borderTop: "1px solid var(--border)",
              }}>
                <span style={{ fontSize: 11, color: "var(--text-muted)", fontStyle: "italic" }}>
                  {turn.response.answer.disclaimer}
                </span>
                <div style={{ display: "flex", gap: 6, flexShrink: 0 }}>
                  <button onClick={() => handleFeedback("up")} style={{
                    background: fb === "up" ? "var(--accent)" : "var(--bg-subtle)",
                    border: "1px solid var(--border)", borderRadius: 6,
                    padding: "4px 8px", cursor: fb ? "default" : "pointer",
                    color: fb === "up" ? "#fff" : "var(--text-muted)",
                    display: "flex", alignItems: "center", gap: 4, fontSize: 12,
                  }}>
                    <ThumbsUp size={12} /> Helpful
                  </button>
                  <button onClick={() => handleFeedback("down")} style={{
                    background: fb === "down" ? "#ef4444" : "var(--bg-subtle)",
                    border: "1px solid var(--border)", borderRadius: 6,
                    padding: "4px 8px", cursor: fb ? "default" : "pointer",
                    color: fb === "down" ? "#fff" : "var(--text-muted)",
                    display: "flex", alignItems: "center", gap: 4, fontSize: 12,
                  }}>
                    <ThumbsDown size={12} />
                  </button>
                </div>
              </div>
            </div>

            {/* Retrieval Trace */}
            <div style={{
              marginTop: 8,
              background: "var(--bg-card)", border: "1px solid var(--border)",
              borderRadius: 10,
            }}>
              <button
                onClick={() => setTraceOpen(o => !o)}
                style={{
                  width: "100%", display: "flex", alignItems: "center",
                  justifyContent: "space-between", padding: "10px 16px",
                  background: "transparent", border: "none",
                  color: "var(--text-muted)", cursor: "pointer", fontSize: 12, fontWeight: 600,
                }}
              >
                <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                  <span style={{ fontFamily: "monospace", color: "var(--accent)" }}>⟨/⟩</span>
                  Retrieval Trace
                  <span style={{
                    fontSize: 10, fontFamily: "monospace", background: "var(--bg-subtle)",
                    border: "1px solid var(--border)", borderRadius: 4, padding: "1px 5px",
                  }}>
                    {turn.response.attempts} attempt{turn.response.attempts !== 1 ? "s" : ""}
                  </span>
                </div>
                {traceOpen ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
              </button>

              {traceOpen && (
                <div style={{ padding: "0 16px 16px", display: "flex", flexDirection: "column", gap: 12 }}>
                  <div style={{ background: "var(--bg-subtle)", borderRadius: 8, padding: 12 }}>
                    <div style={{ fontSize: 10, color: "var(--text-muted)", fontFamily: "monospace", marginBottom: 6, textTransform: "uppercase", letterSpacing: "0.05em" }}>
                      Query rewrite
                    </div>
                    <div style={{ fontSize: 12 }}>
                      <span style={{ color: "var(--text-muted)", fontFamily: "monospace" }}>original → </span>
                      {turn.response.original_query}
                    </div>
                    <div style={{ fontSize: 12, marginTop: 4 }}>
                      <span style={{ color: "var(--accent)", fontFamily: "monospace" }}>rewritten → </span>
                      {turn.response.rewritten_query}
                    </div>
                  </div>

                  <div>
                    <div style={{ fontSize: 10, color: "var(--text-muted)", fontFamily: "monospace", marginBottom: 8, textTransform: "uppercase", letterSpacing: "0.05em" }}>
                      Top retrieved documents
                    </div>
                    {turn.response.retrieved_docs.map((doc, i) => {
                      const { pct, color } = confidenceBar(doc.score);
                      return (
                        <div key={i} style={{
                          background: "var(--bg)", border: "1px solid var(--border)",
                          borderRadius: 8, padding: 12, marginBottom: 8,
                        }}>
                          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                            <span style={{ fontSize: 11, fontFamily: "monospace", color: "var(--amber)" }}>
                              {doc.source} · p.{doc.page}
                            </span>
                            <span style={{ fontSize: 10, fontFamily: "monospace", color }}>{doc.score.toFixed(3)}</span>
                          </div>
                          <div style={{ height: 2, background: "var(--bg-subtle)", borderRadius: 1, marginBottom: 8 }}>
                            <div style={{ height: "100%", width: `${pct}%`, background: color, borderRadius: 1 }} />
                          </div>
                          <p style={{ fontSize: 11, color: "var(--text-muted)", lineHeight: 1.6, fontFamily: "monospace" }}>
                            {doc.excerpt}…
                          </p>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
}

export default function Home() {
  const [turns, setTurns]       = useState<Turn[]>([]);
  const [input, setInput]       = useState("");
  const [taxYear, setTaxYear]   = useState(2025);
  const [loading, setLoading]   = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [turns]);

  const chatHistory = (): ChatMessage[] =>
    turns
      .filter(t => t.response)
      .flatMap(t => [
        { role: "user" as const,      content: t.question },
        { role: "assistant" as const, content: t.response!.answer.answer },
      ]);

  async function handleAsk(q = input) {
    if (!q.trim() || loading) return;
    setInput("");
    setLoading(true);

    const id = crypto.randomUUID();
    const newTurn: Turn = {
      id, question: q, status: "rewriting",
      statusText: "Rewriting query…",
      response: null, error: null, traceOpen: false, feedback: null,
    };
    setTurns(prev => [...prev, newTurn]);

    const update = (patch: Partial<Turn>) =>
      setTurns(prev => prev.map(t => t.id === id ? { ...t, ...patch } : t));

    try {
      const history = chatHistory();
      for await (const ev of askStream(q, taxYear, history)) {
        if (ev.event === "rewriting")  update({ status: "rewriting",  statusText: STATUS_LABELS.rewriting });
        if (ev.event === "retrieving") update({ status: "retrieving", statusText: STATUS_LABELS.retrieving });
        if (ev.event === "evaluating") update({ status: "evaluating", statusText: STATUS_LABELS.evaluating });
        if (ev.event === "generating") update({ status: "generating", statusText: STATUS_LABELS.generating });
        if (ev.event === "done")       update({ status: "idle", response: ev.data as AskResponse });
        if (ev.event === "error")      update({ status: "idle", error: (ev.data as { message: string }).message });
      }
    } catch (e: unknown) {
      update({ status: "idle", error: e instanceof Error ? e.message : "Unexpected error" });
    } finally {
      setLoading(false);
    }
  }

  function handleReset() {
    setTurns([]);
    setInput("");
    setLoading(false);
  }

  return (
    <div style={{ minHeight: "100dvh", background: "var(--bg)", display: "flex", flexDirection: "column" }}>

      {/* Header */}
      <header style={{
        borderBottom: "1px solid var(--border)", padding: "0 20px", height: 52,
        display: "flex", alignItems: "center", justifyContent: "space-between",
        position: "sticky", top: 0, background: "var(--bg)", zIndex: 10,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span style={{ fontSize: 18 }}>⚖️</span>
          <span style={{ fontWeight: 700, fontSize: 16, letterSpacing: "-0.3px" }}>TaxCounsel</span>
          <span style={{
            fontSize: 10, fontFamily: "monospace",
            background: "var(--bg-subtle)", border: "1px solid var(--border)",
            color: "var(--text-muted)", borderRadius: 4, padding: "2px 6px",
          }}>
            8 IRS Pubs · {taxYear}
          </span>
        </div>
        <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
          {turns.length > 0 && (
            <button onClick={handleReset} style={{
              background: "var(--bg-subtle)", border: "1px solid var(--border)",
              color: "var(--text-muted)", borderRadius: 8, padding: "5px 10px",
              cursor: "pointer", fontSize: 12, display: "flex", alignItems: "center", gap: 5,
            }}>
              <RotateCcw size={12} /> New chat
            </button>
          )}
          <ThemeToggle />
        </div>
      </header>

      {/* Chat area */}
      <main style={{ flex: 1, maxWidth: 760, margin: "0 auto", width: "100%", padding: "24px 16px 160px" }}>

        {turns.length === 0 && (
          <div style={{ textAlign: "center", paddingTop: 60 }}>
            <h1 style={{
              fontSize: "clamp(24px, 5vw, 36px)", fontWeight: 800, letterSpacing: "-0.8px",
              background: "linear-gradient(135deg, var(--accent), var(--accent-2))",
              WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
              marginBottom: 10,
            }}>
              Ask your US tax question
            </h1>
            <p style={{ color: "var(--text-muted)", fontSize: 14, marginBottom: 32 }}>
              Answers grounded in IRS publications · page-level citations · multi-turn conversation
            </p>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 8, justifyContent: "center" }}>
              {STARTERS.map(q => (
                <button key={q} onClick={() => { setInput(q); handleAsk(q); }} style={{
                  background: "var(--bg-card)", border: "1px solid var(--border)",
                  color: "var(--text-muted)", borderRadius: 20, padding: "6px 14px",
                  fontSize: 13, cursor: "pointer",
                }}>
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}

        <div style={{ display: "flex", flexDirection: "column", gap: 28 }}>
          {turns.map(turn => (
            <TurnCard key={turn.id} turn={turn} taxYear={taxYear} />
          ))}
        </div>
        <div ref={bottomRef} />
      </main>

      {/* Input bar — fixed bottom */}
      <div style={{
        position: "fixed", bottom: 0, left: 0, right: 0,
        background: "var(--bg)", borderTop: "1px solid var(--border)",
        padding: "12px 16px 20px",
      }}>
        <div style={{ maxWidth: 760, margin: "0 auto" }}>
          <div style={{
            background: "var(--bg-card)", border: "1px solid var(--border)",
            borderRadius: 12, padding: "10px 14px",
            display: "flex", gap: 10, alignItems: "flex-end",
          }}>
            <textarea
              rows={2}
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={e => {
                if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleAsk(); }
              }}
              placeholder="Ask a follow-up or a new tax question… (Shift+Enter for newline)"
              disabled={loading}
              style={{
                flex: 1, resize: "none", background: "transparent",
                border: "none", outline: "none", color: "var(--text)",
                fontSize: 14, lineHeight: 1.6, fontFamily: "inherit",
                opacity: loading ? 0.5 : 1,
              }}
            />
            <div style={{ display: "flex", gap: 8, alignItems: "center", flexShrink: 0 }}>
              <select
                value={taxYear}
                onChange={e => setTaxYear(Number(e.target.value))}
                style={{
                  background: "var(--bg-subtle)", border: "1px solid var(--border)",
                  color: "var(--text)", borderRadius: 6, padding: "4px 6px",
                  fontSize: 12, cursor: "pointer",
                }}
              >
                {YEAR_OPTIONS.map(y => <option key={y} value={y}>{y}</option>)}
              </select>
              <button
                onClick={() => handleAsk()}
                disabled={!input.trim() || loading}
                style={{
                  background: input.trim() && !loading
                    ? "linear-gradient(135deg, var(--accent), var(--accent-2))"
                    : "var(--bg-subtle)",
                  color: input.trim() && !loading ? "#fff" : "var(--text-muted)",
                  border: "none", borderRadius: 8, padding: "8px 14px",
                  fontSize: 13, fontWeight: 600,
                  cursor: input.trim() && !loading ? "pointer" : "not-allowed",
                  display: "flex", alignItems: "center", gap: 5,
                }}
              >
                <Send size={13} /> Ask
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
