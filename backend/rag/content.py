from typing import Any, List


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
