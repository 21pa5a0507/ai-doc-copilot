function getHostLabel(url) {
  if (!url) return "Internal source";

  try {
    return new URL(url).hostname.replace(/^www\./, "");
  } catch {
    return "Internal source";
  }
}

export default function SourcesPanel({ sources }) {
  if (!sources || sources.length === 0) {
    return (
      <div className="flex h-full min-h-0 items-center justify-center p-6">
        <div className="mx-auto max-w-sm text-center">
          <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-2xl border border-[var(--border-strong)] bg-[var(--surface-soft)] text-sm font-semibold text-[var(--text-primary)]">
            SRC
          </div>
          <p className="mt-5 font-display text-2xl font-semibold text-[var(--text-primary)]">
            No sources yet
          </p>
          <p className="mt-3 text-sm leading-6 text-[var(--text-muted)]">
            Matching documents and excerpts will appear here after you ask a question.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-full min-h-0 flex-col bg-transparent">
      <div className="border-b border-[var(--border)] px-6 py-5">
        <div className="flex items-center justify-between gap-3">
          <div>
            <p className="font-display text-xl font-semibold text-[var(--text-primary)]">Sources</p>
            <p className="mt-1 text-sm text-[var(--text-muted)]">
              {sources.length} related source{sources.length === 1 ? "" : "s"}
            </p>
          </div>
          <span className="source-chip">Results</span>
        </div>
      </div>

      <div className="chat-scroll flex-1 space-y-4 overflow-y-auto p-6 min-h-0">
        {sources.map((src, index) => (
          <div
            key={index}
            className="panel-soft space-y-4 rounded-[1.5rem] p-5"
          >
            <div className="flex items-start justify-between gap-3">
              <div className="min-w-0">
                <div className="flex flex-wrap items-center gap-2">
                  <p className="text-xs uppercase tracking-[0.24em] text-[var(--text-muted)]">
                    Source {index + 1}
                  </p>
                  {src.kb_source_label && (
                    <span className="source-chip">{src.kb_source_label}</span>
                  )}
                </div>
                <a
                  href={src.url}
                  target="_blank"
                  rel="noreferrer"
                  className="mt-2 block text-base font-semibold leading-6 text-[var(--text-primary)] transition-colors hover:text-[var(--accent)]"
                >
                  {src.title || "Untitled Source"}
                </a>
                <p className="mt-2 break-all text-xs leading-5 text-[var(--text-muted)]">
                  {getHostLabel(src.url)}{src.url ? ` · ${src.url}` : ""}
                </p>
              </div>

              {typeof src.score === "number" && (
                <div className="source-chip source-chip-score">
                  {src.score.toFixed(2)}
                </div>
              )}
            </div>

            <div className="rounded-2xl border border-[var(--border)] bg-[var(--surface-muted)] px-4 py-4">
              <p className="max-h-48 overflow-y-auto text-sm leading-7 text-[var(--text-secondary)]">
                {src.content
                  ? `${src.content.slice(0, 240)}${src.content.length > 240 ? "..." : ""}`
                  : "No preview available."}
              </p>
            </div>

            {typeof src.score === "number" && (
              <div>
                <div className="mb-2 flex items-center justify-between text-xs text-[var(--text-muted)]">
                  <span>Relevance</span>
                  <span>{Math.round(src.score * 100)}%</span>
                </div>
                <div className="h-2.5 rounded-full bg-[var(--surface-muted)]">
                  <div
                    className="h-2.5 rounded-full bg-[var(--accent)]"
                    style={{ width: `${Math.max(8, Math.min(100, src.score * 100))}%` }}
                  />
                </div>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
