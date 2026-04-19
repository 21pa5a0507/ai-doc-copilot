function contentToText(content) {
  if (typeof content === "string") {
    return content;
  }

  if (content == null) {
    return "";
  }

  if (Array.isArray(content)) {
    return content
      .map((item) => {
        if (typeof item === "string") {
          return item;
        }

        if (item && typeof item === "object") {
          if (typeof item.text === "string") {
            return item.text;
          }

          if (typeof item.content === "string") {
            return item.content;
          }
        }

        return "";
      })
      .filter(Boolean)
      .join("\n\n");
  }

  if (typeof content === "object") {
    if (typeof content.text === "string") {
      return content.text;
    }

    if (typeof content.content === "string") {
      return content.content;
    }
  }

  return String(content);
}

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

const isHeading = (line) => /^###{0,1}\s+/.test(line.trim());
const isBulletLine = (line) => /^[-*•]\s+/.test(line.trim()) || line.trim() === "•";
const isNumberedLine = (line) => /^\d+[.)]\s+/.test(line.trim());
const isDividerLine = (line) => /^([*-])\1{0,}$/.test(line.trim());

function renderInlineFormatting(text) {
  if (!text) {
    return text;
  }

  const parts = [];
  const boldPattern = /\*\*(.*?)\*\*/g;
  let lastIndex = 0;
  let keyIndex = 0;

  for (const match of text.matchAll(boldPattern)) {
    if (match.index > lastIndex) {
      parts.push(<span key={`text-${keyIndex++}`}>{text.slice(lastIndex, match.index)}</span>);
    }

    parts.push(
      <strong key={`bold-${keyIndex++}`} className="font-semibold text-[var(--text-primary)]">
        {match[1]}
      </strong>
    );

    lastIndex = match.index + match[0].length;
  }

  if (parts.length === 0) {
    return text;
  }

  if (lastIndex < text.length) {
    parts.push(<span key={`text-${keyIndex++}`}>{text.slice(lastIndex)}</span>);
  }

  return parts;
}

function collectListItems(lines, startIndex, matcher, cleanItem) {
  const items = [];
  let lineIndex = startIndex;

  while (lineIndex < lines.length && matcher(lines[lineIndex])) {
    const result = cleanItem(lines[lineIndex].trim(), lines, lineIndex);
    if (result?.text) {
      items.push(result.text);
    }
    lineIndex += result?.linesConsumed ?? 1;
  }

  return {
    items,
    nextIndex: lineIndex,
  };
}

function renderMessageContent(content) {
  if (!content || typeof content !== "string") {
    return null;
  }

  const lines = content
    .replace(/\r\n/g, "\n")
    .split("\n")
    .filter((line) => !isDividerLine(line));
  const elements = [];
  let elementIndex = 0;
  let lineIndex = 0;

  while (lineIndex < lines.length) {
    const trimmedLine = lines[lineIndex].trim();

    if (!trimmedLine) {
      elements.push(<div key={`space-${elementIndex++}`} className="h-2" />);
      lineIndex += 1;
      continue;
    }

    if (trimmedLine.startsWith("###") || trimmedLine.startsWith("##")) {
      const level = trimmedLine.startsWith("###") ? "h3" : "h2";
      const heading = trimmedLine.replace(/^###{0,1}\s*/, "").trim();

      if (heading) {
        const Tag = level;
        elements.push(
          <Tag
            key={`${level}-${elementIndex++}`}
            className={
              level === "h3"
                ? "mt-3 mb-2 font-display text-base font-bold text-[var(--text-primary)]"
                : "mt-4 mb-2 font-display text-lg font-bold text-[var(--text-primary)]"
            }
          >
            {renderInlineFormatting(heading)}
          </Tag>
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
                    {renderInlineFormatting(header)}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((row, rowIndex) => (
                <tr key={`tr-${rowIndex}`} className="bg-[var(--surface-elevated)]">
                  {headers.map((_, columnIndex) => (
                    <td
                      key={`td-${rowIndex}-${columnIndex}`}
                      className="border-b border-[var(--border)] px-4 py-3 align-top text-[var(--text-secondary)] last:border-b-0"
                    >
                      {renderInlineFormatting(row[columnIndex] || "-")}
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
      const { items, nextIndex } = collectListItems(
        lines,
        lineIndex,
        isBulletLine,
        (line, sourceLines, currentIndex) => {
          let item = line === "•" ? "" : line.replace(/^[-*•]\s*/, "").trim();
          let linesConsumed = 1;

          if (!item && currentIndex + 1 < sourceLines.length) {
            const nextLine = sourceLines[currentIndex + 1].trim();
            if (nextLine && !isHeading(nextLine) && !isBulletLine(nextLine) && !isNumberedLine(nextLine)) {
              item = nextLine;
              linesConsumed = 2;
            }
          }

          return { text: item, linesConsumed };
        }
      );

      if (items.length > 0) {
        elements.push(
          <ul
            key={`ul-${elementIndex++}`}
            className="mb-3 list-disc space-y-2 pl-5 text-[var(--text-secondary)] marker:text-[var(--accent)]"
          >
            {items.map((item, index) => (
              <li key={`bullet-${index}`} className="leading-relaxed">
                {renderInlineFormatting(item)}
              </li>
            ))}
          </ul>
        );
      }

      lineIndex = nextIndex;
      continue;
    }

    if (isNumberedLine(trimmedLine)) {
      const { items, nextIndex } = collectListItems(
        lines,
        lineIndex,
        isNumberedLine,
        (line) => ({ text: line.replace(/^\d+[.)]\s*/, "").trim(), linesConsumed: 1 })
      );

      if (items.length > 0) {
        elements.push(
          <ol
            key={`ol-${elementIndex++}`}
            className="mb-3 list-decimal space-y-2 pl-5 text-[var(--text-secondary)] marker:text-[var(--accent)]"
          >
            {items.map((item, index) => (
              <li key={`number-${index}`} className="leading-relaxed">
                {renderInlineFormatting(item)}
              </li>
            ))}
          </ol>
        );
      }

      lineIndex = nextIndex;
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

    elements.push(
      <p key={`p-${elementIndex++}`} className="mb-3 leading-7 text-[var(--text-secondary)]">
        {renderInlineFormatting(paragraphLines.join(" "))}
      </p>
    );
  }

  return elements.length > 0 ? elements : null;
}

export { contentToText, renderMessageContent };
