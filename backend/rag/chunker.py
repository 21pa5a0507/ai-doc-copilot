import re

def split_paragraphs(text):
    return [p.strip() for p in re.split(r'\n+', text) if p.strip()]

def split_sentences(text):
    return re.split(r'(?<=[.!?])\s+', text)

def get_overlap_text(text, overlap_words=30):
    words = text.split()
    return " ".join(words[-overlap_words:]) if len(words) > overlap_words else text

def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()

def chunk_text(text, chunk_size=600, overlap_words=150, min_chunk_size=100):
    text = clean_text(text)
    paragraphs = split_paragraphs(text)

    chunks = []
    current_chunk = ""

    for para in paragraphs:

        # ✅ Keep small paragraphs (merge instead of removing)
        if len(para) < 40:
            current_chunk += " " + para
            continue

        # ✅ Handle large paragraphs
        if len(para) > chunk_size:
            sentences = split_sentences(para)

            for sentence in sentences:
                if len(current_chunk) + len(sentence) <= chunk_size:
                    current_chunk += " " + sentence
                else:
                    if len(current_chunk.strip()) >= min_chunk_size:
                        chunks.append(current_chunk.strip())

                    overlap_text = get_overlap_text(current_chunk, overlap_words)
                    current_chunk = overlap_text + " " + sentence

            continue

        # ✅ Normal case
        if len(current_chunk) + len(para) <= chunk_size:
            current_chunk += " " + para
        else:
            if len(current_chunk.strip()) >= min_chunk_size:
                chunks.append(current_chunk.strip())

            overlap_text = get_overlap_text(current_chunk, overlap_words)
            current_chunk = overlap_text + " " + para

    # ✅ Final chunk
    if len(current_chunk.strip()) >= min_chunk_size:
        chunks.append(current_chunk.strip())

    return chunks