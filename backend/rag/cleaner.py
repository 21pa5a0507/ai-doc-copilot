import re
from bs4 import BeautifulSoup
import markdown2


def clean_text(text):
    if not text:
        return ""

    # Convert Markdown to HTML before plain text extraction
    html = markdown2.markdown(text)
    soup = BeautifulSoup(html, "lxml")

    # Remove script/style and navigation boilerplate
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    plain = soup.get_text(" ")

    # Normalize whitespace and remove common artifacts
    plain = re.sub(r"\s+", " ", plain).strip()

    # Remove inline code markers, excessive headers or lists markers
    plain = re.sub(r"`([^`]+)`", r"\1", plain)
    plain = re.sub(r"\*\*|\*|__", "", plain)

    # Remove common junk tokens from docs scraping
    junk_patterns = [
        r"\btheme auto light dark\b",
        r"\btable of contents\b",
        r"\bnavigation\b",
        r"\bprevious\b",
        r"\bnext\b",
        r"\bedit this page\b",
        r"\bgithub\b",
        r"\* \* \*", r"^-{3,}$"
    ]

    for p in junk_patterns:
        plain = re.sub(p, "", plain, flags=re.I)

    plain = plain.replace("✨", "")

    # Final trim and collapse whitespace
    plain = re.sub(r"\s+", " ", plain).strip()

    return plain