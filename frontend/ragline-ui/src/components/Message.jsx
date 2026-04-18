import React from "react";

/**
 * Message Component - Renders chat messages with formatted content support
 * 
 * Supported Formatting for AI Messages:
 * - ** text ** = Bold text (both ** work)
 * - ## Heading = Large heading
 * - ### Heading = Medium heading
 * - - Item, * Item, or • Item = Bullet points
 * - 1. Item or 1) Item = Numbered points
 * - Line breaks automatically create new paragraphs
 * 
 * Example:
 * "## Overview\nThis is a **bold** concept.\n- Point 1\n- Point 2\n### Details\nMore info here"
 */
export default function Message({ message }) {
  const { role, text } = message;

  const parseTableRow = (line) =>
    line
      .trim()
      .replace(/^\|/, "")
      .replace(/\|$/, "")
      .split("|")
      .map((cell) => cell.trim());

  const isTableSeparator = (line) =>
    /^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?$/.test(line.trim());

  const isTableRow = (line) => {
    const trimmed = line.trim();
    return trimmed.includes("|") && parseTableRow(trimmed).length > 1;
  };

  const parseContent = (content) => {
    if (!content || typeof content !== "string") return null;

    const lines = content.replace(/\r\n/g, "\n").split("\n");
    const elements = [];
    let elementIndex = 0;
    let lineIndex = 0;

    const isHeading = (line) => /^###{0,1}\s+/.test(line.trim());
    const isBulletLine = (line) => /^[-*•]\s+/.test(line.trim()) || line.trim() === "•";
    const isNumberedLine = (line) => /^\d+[.)]\s+/.test(line.trim());

    while (lineIndex < lines.length) {
      const line = lines[lineIndex];
      const trimmedLine = line.trim();

      if (!trimmedLine) {
        elements.push(<div key={`empty-${elementIndex++}`} className="h-2" />);
        lineIndex += 1;
        continue;
      }

      if (trimmedLine.startsWith("###")) {
        const heading = trimmedLine.replace(/^###\s*/, "").trim();
        if (heading) {
          elements.push(
            <h3
              key={`h3-${elementIndex++}`}
              className="mt-3 mb-2 font-display text-base font-bold text-[var(--text-primary)]"
            >
              {parseInlineFormatting(heading)}
            </h3>
          );
        }
        lineIndex += 1;
        continue;
      }

      if (trimmedLine.startsWith("##")) {
        const heading = trimmedLine.replace(/^##\s*/, "").trim();
        if (heading) {
          elements.push(
            <h2
              key={`h2-${elementIndex++}`}
              className="mt-4 mb-2 font-display text-lg font-bold text-[var(--text-primary)]"
            >
              {parseInlineFormatting(heading)}
            </h2>
          );
        }
        lineIndex += 1;
        continue;
      }

      if (
        lineIndex + 1 < lines.length &&
        isTableRow(trimmedLine) &&
        isTableSeparator(lines[lineIndex + 1])
      ) {
        const headers = parseTableRow(trimmedLine);
        const rows = [];
        lineIndex += 2;

        while (lineIndex < lines.length && isTableRow(lines[lineIndex])) {
          rows.push(parseTableRow(lines[lineIndex]));
          lineIndex += 1;
        }

        elements.push(
          <div
            key={`table-${elementIndex++}`}
            className="mb-4 overflow-x-auto rounded-[1rem] border border-[var(--border)]"
          >
            <table className="min-w-full border-collapse text-left text-sm">
              <thead className="bg-[var(--surface-muted)]">
                <tr>
                  {headers.map((header, index) => (
                    <th
                      key={`th-${index}`}
                      className="border-b border-[var(--border)] px-4 py-3 font-semibold text-[var(--text-primary)]"
                    >
                      {parseInlineFormatting(header)}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.map((row, rowIndex) => (
                  <tr key={`tr-${rowIndex}`} className="bg-[var(--surface-elevated)]">
                    {headers.map((_, colIndex) => (
                      <td
                        key={`td-${rowIndex}-${colIndex}`}
                        className="border-b border-[var(--border)] px-4 py-3 align-top text-[var(--text-secondary)] last:border-b-0"
                      >
                        {parseInlineFormatting(row[colIndex] || "-")}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        );
        continue;
      }

      if (isBulletLine(trimmedLine)) {
        const items = [];

        while (lineIndex < lines.length && isBulletLine(lines[lineIndex])) {
          let currentLine = lines[lineIndex].trim();
          let item = currentLine === "•" ? "" : currentLine.replace(/^[-*•]\s*/, "").trim();

          if (!item && lineIndex + 1 < lines.length) {
            const nextLine = lines[lineIndex + 1].trim();
            if (nextLine && !isHeading(nextLine) && !isBulletLine(nextLine) && !isNumberedLine(nextLine)) {
              item = nextLine;
              lineIndex += 1;
            }
          }

          if (item) {
            items.push(item);
          }
          lineIndex += 1;
        }

        if (items.length > 0) {
          elements.push(
            <ul
              key={`ul-${elementIndex++}`}
              className="mb-3 list-disc space-y-2 pl-5 text-[var(--text-secondary)] marker:text-[var(--accent)]"
            >
              {items.map((item, index) => (
                <li key={`bullet-${index}`} className="leading-relaxed">
                  {parseInlineFormatting(item)}
                </li>
              ))}
            </ul>
          );
        }
        continue;
      }

      if (isNumberedLine(trimmedLine)) {
        const items = [];

        while (lineIndex < lines.length && isNumberedLine(lines[lineIndex])) {
          const item = lines[lineIndex].trim().replace(/^\d+[.)]\s*/, "").trim();
          if (item) {
            items.push(item);
          }
          lineIndex += 1;
        }

        if (items.length > 0) {
          elements.push(
            <ol
              key={`ol-${elementIndex++}`}
              className="mb-3 list-decimal space-y-2 pl-5 text-[var(--text-secondary)] marker:text-[var(--accent)]"
            >
              {items.map((item, index) => (
                <li key={`number-${index}`} className="leading-relaxed">
                  {parseInlineFormatting(item)}
                </li>
              ))}
            </ol>
          );
        }
        continue;
      }

      const paragraphLines = [trimmedLine];
      lineIndex += 1;

      while (lineIndex < lines.length) {
        const nextLine = lines[lineIndex].trim();
        if (!nextLine || isHeading(nextLine) || isBulletLine(nextLine) || isNumberedLine(nextLine)) {
          break;
        }
        paragraphLines.push(nextLine);
        lineIndex += 1;
      }

      const paragraph = paragraphLines.join(" ");
      if (paragraph) {
        elements.push(
          <p key={`p-${elementIndex++}`} className="mb-3 leading-7 text-[var(--text-secondary)]">
            {parseInlineFormatting(paragraph)}
          </p>
        );
      }
    }

    return elements.length > 0 ? elements : null;
  };

  const parseInlineFormatting = (text) => {
    if (!text) return text;
    
    const parts = [];
    let lastIndex = 0;
    let partIndex = 0;

    // Match ** for bold
    const boldRegex = /\*\*(.*?)\*\*/g;
    const matches = Array.from(text.matchAll(boldRegex));

    if (matches.length === 0) {
      return text;
    }

    matches.forEach((boldMatch) => {
      // Add text before bold
      if (boldMatch.index > lastIndex) {
        parts.push(
          <React.Fragment key={`text-${partIndex++}`}>
            {text.substring(lastIndex, boldMatch.index)}
          </React.Fragment>
        );
      }

      // Add bold text
      parts.push(
        <strong key={`bold-${partIndex++}`} className="font-semibold text-[var(--text-primary)]">
          {boldMatch[1]}
        </strong>
      );

      lastIndex = boldMatch.index + boldMatch[0].length;
    });

    // Add remaining text
    if (lastIndex < text.length) {
      parts.push(
        <React.Fragment key={`text-${partIndex++}`}>
          {text.substring(lastIndex)}
        </React.Fragment>
      );
    }

    return parts.length > 0 ? parts : text;
  };

  return (
    <div className={`message-shell flex items-start gap-3 ${role === "user" ? "justify-end" : "justify-start"}`}>
      {role === "ai" && (
        <div className="message-avatar message-avatar-ai">
          HP
        </div>
      )}

      <div
        className={`w-full max-w-[42rem] sm:w-[92%] lg:w-[86%] xl:w-[78%] rounded-[1.35rem] border px-4 py-3 shadow-[var(--shadow-soft)] ${
          role === "user"
            ? "border-transparent bg-[var(--accent)] text-white"
            : "border-[var(--border)] bg-[var(--surface-elevated)] text-[var(--text-primary)]"
        }`}
      >
        {role === "ai" ? (
          parseContent(text)
        ) : (
          <span className="whitespace-pre-wrap leading-7 text-white/95">{text}</span>
        )}
      </div>

      {role === "user" && (
        <div className="message-avatar message-avatar-user">
          Y
        </div>
      )}
    </div>
  );
}
