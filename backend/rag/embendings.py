from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
_model = None


def _get_model():
    global _model

    if _model is None:
        print(f"Loading embedding model: {MODEL_NAME}")
        _model = SentenceTransformer(MODEL_NAME)

    return _model

def get_embending(texts):
    """
    Supports both single string and list of strings
    """

    # Convert to list if single input
    single_input = False
    if isinstance(texts, str):
        texts = [texts]  # Add query prefix for single string input
        single_input = True

    model = _get_model()

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
