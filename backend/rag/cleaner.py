import re
from bs4 import BeautifulSoup


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


def clean_text(raw_html: str) -> str:
    if not raw_html:
        return ""

    soup = BeautifulSoup(raw_html, "html.parser")

    for tag in soup([
        "script", "style", "nav", "footer", "header",
        "aside", "noscript"
    ]):
        tag.decompose()

    text = soup.get_text(separator="\n")

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
