"use client";

import { useCallback, useState, type CSSProperties } from "react";
import Markdown from "react-markdown";

export type LexaiCitation = {
  label: string;
  chunk_id?: number;
  snippet?: string;
};

export type LexaiChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations?: LexaiCitation[];
  streaming?: boolean;
};

export type LexaiChatThreadProps = {
  messages: LexaiChatMessage[];
  onFeedback?: (messageId: string, vote: "up" | "down") => void;
};

function SourceBadge({ label }: { label: string }) {
  return (
    <span
      style={{
        display: "inline-block",
        marginTop: 6,
        marginRight: 6,
        padding: "4px 8px",
        borderRadius: 8,
        fontSize: 12,
        fontWeight: 600,
        background: "rgba(108, 140, 255, 0.18)",
        border: "1px solid rgba(108, 140, 255, 0.35)",
        color: "#e8ecf5",
      }}
    >
      [{label}]
    </span>
  );
}

function StreamingDots() {
  return (
    <span className="lexai-stream-dots" aria-hidden>
      <span className="lexai-dot">.</span>
      <span className="lexai-dot">.</span>
      <span className="lexai-dot">.</span>
      <style>{`
        @keyframes lexaiBlink {
          0%, 80%, 100% { opacity: 0.2; }
          40% { opacity: 1; }
        }
        .lexai-stream-dots { display: inline-flex; gap: 2px; margin-left: 6px; }
        .lexai-dot { animation: lexaiBlink 1.2s infinite ease-in-out both; font-weight: 700; }
        .lexai-dot:nth-child(2) { animation-delay: 0.15s; }
        .lexai-dot:nth-child(3) { animation-delay: 0.3s; }
      `}</style>
    </span>
  );
}

const btnStyle: CSSProperties = {
  fontSize: 12,
  padding: "4px 8px",
  borderRadius: 8,
  border: "1px solid rgba(255,255,255,0.18)",
  background: "rgba(0,0,0,0.35)",
  color: "#e8ecf5",
  cursor: "pointer",
};

export function LexaiChatThread({ messages, onFeedback }: LexaiChatThreadProps) {
  const [copiedId, setCopiedId] = useState<string | null>(null);

  const copy = useCallback(async (id: string, text: string) => {
    await navigator.clipboard.writeText(text);
    setCopiedId(id);
    window.setTimeout(() => setCopiedId(null), 1600);
  }, []);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14, padding: 8 }}>
      {messages.map((m) => {
        const isUser = m.role === "user";
        return (
          <div
            key={m.id}
            style={{
              alignSelf: isUser ? "flex-end" : "flex-start",
              maxWidth: "min(760px, 94%)",
            }}
          >
            <div
              style={{
                borderRadius: 14,
                padding: "12px 14px",
                border: "1px solid rgba(255,255,255,0.12)",
                background: isUser ? "rgba(108,140,255,0.18)" : "rgba(255,255,255,0.05)",
                color: "#e8ecf5",
                lineHeight: 1.45,
              }}
            >
              {isUser ? (
                <>
                  <div style={{ fontSize: 11, opacity: 0.65, marginBottom: 6 }}>Você</div>
                  <div style={{ whiteSpace: "pre-wrap" }}>{m.content}</div>
                </>
              ) : (
                <div className="lexai-assistant-wrap" style={{ position: "relative" }}>
                  <div
                    className="lexai-msg-actions"
                    style={{
                      position: "absolute",
                      top: 0,
                      right: 0,
                      display: "flex",
                      gap: 6,
                      opacity: 0,
                      transition: "opacity 0.15s ease",
                      zIndex: 1,
                    }}
                  >
                    <button
                      type="button"
                      title="Copiar"
                      onClick={() => void copy(m.id, m.content)}
                      style={btnStyle}
                    >
                      {copiedId === m.id ? "Copiado" : "Copiar"}
                    </button>
                    <button type="button" title="Útil" onClick={() => onFeedback?.(m.id, "up")} style={btnStyle}>
                      👍
                    </button>
                    <button type="button" title="Não útil" onClick={() => onFeedback?.(m.id, "down")} style={btnStyle}>
                      👎
                    </button>
                  </div>
                  <div style={{ fontSize: 11, opacity: 0.65, marginBottom: 6 }}>Gemini</div>
                  <div className="lexai-md">
                    {m.content ? <Markdown>{m.content}</Markdown> : null}
                    {m.streaming ? <StreamingDots /> : null}
                  </div>
                </div>
              )}
            </div>
            {!isUser && m.citations?.length ? (
              <div style={{ marginTop: 8 }}>
                {m.citations.map((c, i) => (
                  <SourceBadge key={`${m.id}-c-${i}`} label={c.label} />
                ))}
              </div>
            ) : null}
          </div>
        );
      })}
      <style>{`
        .lexai-assistant-wrap:hover .lexai-msg-actions,
        .lexai-assistant-wrap:focus-within .lexai-msg-actions {
          opacity: 1;
        }
      `}</style>
    </div>
  );
}
