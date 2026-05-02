const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export interface TaxAnswer {
  answer: string;
  page_citations: string[];
  tax_year: number;
  confidence: number;
  disclaimer: string;
}

export interface RetrievedDoc {
  source: string;
  section: string;
  page: number;
  score: number;
  excerpt: string;
}

export interface AskResponse {
  answer: TaxAnswer;
  original_query: string;
  rewritten_query: string;
  retrieved_docs: RetrievedDoc[];
  attempts: number;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export type StreamEvent =
  | { event: "rewriting";   data: { query: string; attempt: number } }
  | { event: "retrieving";  data: { query: string } }
  | { event: "evaluating";  data: Record<string, never> }
  | { event: "generating";  data: Record<string, never> }
  | { event: "done";        data: AskResponse }
  | { event: "error";       data: { message: string } };

export async function askQuestion(
  question: string,
  taxYear: number,
  chatHistory: ChatMessage[] = []
): Promise<AskResponse> {
  const res = await fetch(`${API_BASE}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, tax_year: taxYear, chat_history: chatHistory }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail ?? `API error ${res.status}`);
  }
  return res.json();
}

export async function* askStream(
  question: string,
  taxYear: number,
  chatHistory: ChatMessage[] = []
): AsyncGenerator<StreamEvent> {
  const res = await fetch(`${API_BASE}/ask/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, tax_year: taxYear, chat_history: chatHistory }),
  });
  if (!res.ok || !res.body) throw new Error(`Stream error ${res.status}`);

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    const parts = buf.split("\n\n");
    buf = parts.pop() ?? "";
    for (const part of parts) {
      const lines = part.trim().split("\n");
      let eventType = "message";
      let dataStr = "";
      for (const line of lines) {
        if (line.startsWith("event: ")) eventType = line.slice(7);
        if (line.startsWith("data: "))  dataStr   = line.slice(6);
      }
      if (dataStr) {
        try {
          yield { event: eventType, data: JSON.parse(dataStr) } as StreamEvent;
        } catch { /* skip malformed */ }
      }
    }
  }
}

export async function submitFeedback(
  question: string,
  answer: string,
  taxYear: number,
  isHelpful: boolean,
  comment = ""
): Promise<void> {
  await fetch(`${API_BASE}/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question, answer, tax_year: taxYear, is_helpful: isHelpful, comment,
    }),
  });
}
