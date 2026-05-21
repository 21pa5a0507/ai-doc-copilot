import re
from typing import Any, List


JUNK_PATTERNS = [
    r"Table of contents",
    r"Search",
    r"Navigation",
    r"Menu",
    r"Previous",
    r"Next",
    r"Edit on GitHub",
    r"GitHub",
    r"Release Notes",
    r"About",
    r"Resources",
    r"Skip to content",
]


def clean_text(text: str) -> str:
    if not text:
        return ""

    for pattern in JUNK_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    text = re.sub(r"\b(\w+)( \1\b)+", r"\1", text)
    text = re.sub(r"[^\w\s.,:/@()-]", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = text.strip()

    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        line = line.strip()
        if len(line) > 40:
            cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)

    return text


def content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content

    if content is None:
        return ""

    if isinstance(content, list):
        parts: List[str] = []

        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue

            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
                continue

            text = getattr(item, "text", None) or getattr(item, "content", None)
            if isinstance(text, str):
                parts.append(text)

        return "\n".join(part for part in parts if part).strip()

    if isinstance(content, dict):
        text = content.get("text") or content.get("content")
        if isinstance(text, str):
            return text

    text = getattr(content, "text", None) or getattr(content, "content", None)
    if isinstance(text, str):
        return text

    return str(content)
