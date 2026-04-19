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

    # -----------------------------
    # 1. Parse HTML
    # -----------------------------
    soup = BeautifulSoup(raw_html, "html.parser")

    # -----------------------------
    # 2. Remove unwanted tags
    # -----------------------------
    for tag in soup([
        "script", "style", "nav", "footer", "header",
        "aside", "noscript"
    ]):
        tag.decompose()

    # -----------------------------
    # 3. Get visible text
    # -----------------------------
    text = soup.get_text(separator="\n")

    # -----------------------------
    # 4. Remove common UI junk
    # -----------------------------
    for pattern in JUNK_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    # -----------------------------
    # 5. Remove repeated words (e.g., "your your your")
    # -----------------------------
    text = re.sub(r"\b(\w+)( \1\b)+", r"\1", text)

    # -----------------------------
    # 6. Remove excessive symbols / noise
    # -----------------------------
    text = re.sub(r"[^\w\s.,:/@()-]", " ", text)

    # -----------------------------
    # 7. Normalize whitespace
    # -----------------------------
    text = re.sub(r"\n\s*\n+", "\n\n", text)  # collapse empty lines
    text = re.sub(r"[ \t]+", " ", text)       # collapse spaces
    text = text.strip()

    # -----------------------------
    # 8. Remove very short lines
    # -----------------------------
    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        line = line.strip()
        if len(line) > 40:   # threshold (important!)
            cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)

    return text
