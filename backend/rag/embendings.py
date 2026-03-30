from sentence_transformers import SentenceTransformer

# Load model once (VERY IMPORTANT)
model = SentenceTransformer('all-MiniLM-L6-v2')  # 🔥 CRITICAL: load on GPU if available

def get_embending(texts):
    """
    Supports both single string and list of strings
    """

    # Convert to list if single input
    single_input = False
    if isinstance(texts, str):
        texts = [texts]  # Add query prefix for single string input
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