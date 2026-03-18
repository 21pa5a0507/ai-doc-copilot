from sentence_transformers import SentenceTransformer

# Load model once (VERY IMPORTANT)
model = SentenceTransformer('all-MiniLM-L6-v2')


def get_embending(texts):
    """
    Supports both single string and list of strings
    """

    # Convert to list if single input
    single_input = False
    if isinstance(texts, str):
        texts = [texts]
        single_input = True

    embeddings = model.encode(
        texts,
        batch_size=32,              # 🔥 faster
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True  # 🔥 CRITICAL (cosine similarity fix)
    )

    if single_input:
        return embeddings[0].tolist()

    return embeddings.tolist()

"""
import json

from rag.cleaner import clean_text
from rag.chunker import chunk_text
from rag.embendings import get_embending


# -------------------------------
# ✅ Chunk Filtering
# -------------------------------
def is_valid_chunk(text):
    if not text or len(text) < 50:
        return False

    text_lower = text.lower()

    blacklist = [
        "next", "previous", "edit on github",
        "table of contents", "navigation"
    ]

    if any(word in text_lower for word in blacklist):
        return False

    return True


# -------------------------------
# ✅ Chunking Docs
# -------------------------------
def chunking_docs(docs):

    chunked_data = []
    seen = set()   # deduplication

    for doc in docs:

        cleaned = clean_text(doc["content"])
        chunks = chunk_text(cleaned)

        for chunk in chunks:

            if not is_valid_chunk(chunk):
                continue

            if chunk in seen:
                continue

            seen.add(chunk)

            chunked_data.append({
                "url": doc["url"],
                "title": doc["title"],
                "content": chunk
            })

    print(f"✅ Total clean chunks: {len(chunked_data)}")

    return chunked_data


# -------------------------------
# ✅ Embedding Docs (BATCH)
# -------------------------------
def embedding_docs(docs, store):

    texts = [doc["content"] for doc in docs]

    print("🔄 Generating embeddings in batch...")
    embeddings = get_embending(texts)   # MUST support batch

    for doc, emb in zip(docs, embeddings):
        store.add(emb, doc)

    print("✅ Embeddings stored")

    return store


# -------------------------------
# ✅ MAIN PIPELINE
# -------------------------------
async def scrap_website(store):

    # Load pre-scraped docs
    with open("fastapi_docs.json", "r", encoding="utf-8") as f:
        docs = json.load(f)

    print(f"📄 Loaded {len(docs)} documents")

    # Step 1: Chunking + Cleaning + Filtering
    chunked_docs = chunking_docs(docs)

    # Step 2: Embedding + Store
    store = embedding_docs(chunked_docs, store)

    # Step 3: Build BM25 (🔥 IMPORTANT FIX)
    store.build_bm25()

    print("✅ BM25 built successfully")

    return store
"""