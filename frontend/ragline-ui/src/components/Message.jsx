import { contentToText, renderMessageContent } from "../utils/messageContent";

export default function Message({ message }) {
  const { role, text } = message;
  const normalizedText = contentToText(text);
  const isUser = role === "user";

  return (
    <div className={`message-shell flex items-start gap-3 ${isUser ? "justify-end" : "justify-start"}`}>
      {!isUser && (
        <div className="message-avatar message-avatar-ai">
          HP
        </div>
      )}

      <div
        className={`w-full max-w-[42rem] rounded-[1.35rem] border px-4 py-3 shadow-[var(--shadow-soft)] sm:w-[92%] lg:w-[86%] xl:w-[78%] ${
          isUser
            ? "border-transparent bg-[var(--accent)] text-white"
            : "border-[var(--border)] bg-[var(--surface-elevated)] text-[var(--text-primary)]"
        }`}
      >
        {isUser ? (
          <span className="whitespace-pre-wrap leading-7 text-white/95">{normalizedText}</span>
        ) : (
          renderMessageContent(normalizedText)
        )}
      </div>

      {isUser && (
        <div className="message-avatar message-avatar-user">
          Y
        </div>
      )}
    </div>
  );
}
