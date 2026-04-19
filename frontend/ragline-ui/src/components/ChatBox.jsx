import { useState, useEffect, useRef } from "react";
import { askQuestion } from "../api/ragApi";
import Message from "./Message";
import { contentToText } from "../utils/messageContent";

const SOURCE_OPTIONS = [
  {
    value: "default",
    label: "Hexnode Docs",
    hint: "MDM setup guides, actions, policies, and troubleshooting",
  },
  {
    value: "keka",
    label: "Keka Policies",
    hint: "HR policies, leave rules, payroll terms, and internal programs",
  },
  {
    value: "both",
    label: "Both Sources",
    hint: "Combine Hexnode device guidance with Keka HR policies in one grounded answer",
  },
];

const SUGGESTED_PROMPTS = [
  "How to enroll a Windows device via open enrollment?",
  "Explain the Profit-Sharing Program in simple points",
];

export default function ChatBox({ setSources, messages, setMessages, setIsLoadedChat }) {
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [source, setSource] = useState("default");

  const bottomRef = useRef(null);
  const textareaRef = useRef(null);
  const typingIntervalRef = useRef(null);
  const selectedSource = SOURCE_OPTIONS.find((item) => item.value === source) || SOURCE_OPTIONS[0];

  const updateLastMessageText = (text) => {
    setMessages((prev) => {
      if (prev.length === 0) {
        return prev;
      }

      const updated = [...prev];
      updated[updated.length - 1] = {
        ...updated[updated.length - 1],
        text,
      };
      return updated;
    });
  };

  const stopTypingAnimation = () => {
    if (typingIntervalRef.current) {
      clearInterval(typingIntervalRef.current);
      typingIntervalRef.current = null;
    }
  };

  const syncTextareaHeight = () => {
    const textarea = textareaRef.current;
    if (!textarea) return;

    const maxHeight = 256;
    textarea.style.height = "auto";
    textarea.style.height = `${Math.min(textarea.scrollHeight, maxHeight)}px`;
    textarea.style.overflowY = textarea.scrollHeight > maxHeight ? "auto" : "hidden";
  };

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    syncTextareaHeight();
  }, [question]);

  useEffect(() => () => stopTypingAnimation(), []);

  const typeMessage = (text) => {
    const safeText = contentToText(text);

    stopTypingAnimation();

    if (!safeText) {
      updateLastMessageText("");
      return;
    }

    let index = 0;
    let currentText = "";

    typingIntervalRef.current = setInterval(() => {
      currentText += safeText[index];
      index++;

      updateLastMessageText(currentText);

      if (index === safeText.length) {
        stopTypingAnimation();
      }
    }, 15); // speed (lower = faster)
  };

  const handleAsk = async () => {
    const trimmedQuestion = question.trim();
    if (!trimmedQuestion || loading) return;

    const userMessage = { role: "user", text: trimmedQuestion };
    const newMessages = [...messages, userMessage];
    setMessages(newMessages);
    setIsLoadedChat(false);
    setQuestion("");
    setLoading(true);

    try {
      const data = await askQuestion(trimmedQuestion, source);

      setMessages((prev) => {
        const updated = [...prev, { role: "ai", text: "" }];

        setTimeout(() => {
          typeMessage(data.answer);
        }, 100);

        return updated;
      });
      setSources(Array.isArray(data.chunks) ? data.chunks : []);

    } catch (err) {
      console.error(err);
      setMessages((prev) => [...prev, { role: "ai", text: "Sorry, an error occurred." }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex h-full min-h-0 flex-col overflow-hidden">
      <div className="border-b border-[var(--border)] bg-[var(--surface-soft)]/88 px-4 py-3 backdrop-blur-xl sm:px-5">
        <div className="flex flex-col gap-3">
          <div className="flex flex-wrap gap-2">
            {SOURCE_OPTIONS.map((option) => (
              <button
                key={option.value}
                type="button"
                onClick={() => setSource(option.value)}
                className={`source-toggle ${source === option.value ? "source-toggle-active" : ""}`}
              >
                <span className="text-sm font-semibold">{option.label}</span>
              </button>
            ))}
          </div>

          <p className="text-sm leading-6 text-[var(--text-muted)]">
            {selectedSource.hint}
          </p>

          {messages.length === 0 && (
            <div className="flex flex-wrap gap-2">
              {SUGGESTED_PROMPTS.map((prompt) => (
                <button
                  key={prompt}
                  type="button"
                  onClick={() => setQuestion(prompt)}
                  className="suggestion-chip"
                >
                  {prompt}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      <div className="chat-scroll flex-1 overflow-y-auto px-4 py-5 min-h-0 sm:px-5">
        {messages.length === 0 && (
          <div className="empty-state">
            <h2 className="font-display text-3xl font-semibold tracking-tight text-[var(--text-primary)]">
              Ask a question.
            </h2>
            <p className="mt-3 max-w-2xl text-sm leading-7 text-[var(--text-secondary)]">
              Select a source above, ask naturally, and review the related references on the right when needed. Use
              <span className="font-semibold text-[var(--text-primary)]"> Both Sources </span>
              when you want one answer grounded in both Hexnode and Keka.
            </p>
          </div>
        )}

        <div className="flex w-full flex-col gap-4">
          {messages.map((msg, i) => (
            <Message key={i} message={msg} />
          ))}
        </div>

        {loading && (
          <div className="mt-5 flex w-full items-start gap-3">
            <div className="message-avatar message-avatar-ai">
              HP
            </div>
            <div className="panel-soft rounded-[1.5rem] px-4 py-3">
              <div className="flex space-x-1">
                <div className="h-2 w-2 rounded-full bg-[var(--accent)] animate-bounce" />
                <div
                  className="h-2 w-2 rounded-full bg-[var(--accent)] animate-bounce"
                  style={{ animationDelay: "0.1s" }}
                />
                <div
                  className="h-2 w-2 rounded-full bg-[var(--accent)] animate-bounce"
                  style={{ animationDelay: "0.2s" }}
                />
              </div>
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      <div className="border-t border-[var(--border)] bg-[var(--surface-soft)]/92 px-4 py-4 backdrop-blur-xl sm:px-5">
        <div className="composer-shell">
          <div className="min-w-0 flex-1">
            <textarea
              ref={textareaRef}
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  handleAsk();
                }
              }}
              className="composer-input"
              placeholder={`Ask from ${selectedSource.label}...`}
              rows={1}
              onInput={syncTextareaHeight}
            />
          </div>

          <button
            type="button"
            onClick={handleAsk}
            disabled={loading || !question.trim()}
            className="primary-button w-full shrink-0 justify-center disabled:cursor-not-allowed disabled:opacity-50 sm:w-auto sm:min-w-[9.5rem]"
          >
            {loading ? "Thinking..." : "Send"}
          </button>
        </div>
      </div>
    </div>
  );
}
