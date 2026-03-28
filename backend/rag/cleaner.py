import re

def clean_text(text: str) -> str:
    if not text:
        return ""

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # Fix spacing issues
    text = text.replace(" .", ".").replace(" ,", ",")

    # Remove invisible characters
    text = text.replace("\xa0", " ")

    # Optional: remove simple junk phrases
    junk_patterns = [
        r"\btable of contents\b",
        r"\bprevious\b",
        r"\bnext\b",
        r"\bedit this page\b"
    ]

    for pattern in junk_patterns:
        text = re.sub(pattern, "", text, flags=re.I)

    # Final cleanup
    text = re.sub(r"\s+", " ", text).strip()

    return text