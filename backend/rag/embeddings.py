from rag.backends.onnx_embeddings import get_embedding_model

MODEL_NAME = "all-MiniLM-L6-v2"


def get_embedding(texts):
    """
    Supports both a single string and a list of strings.
    """
    single_input = False

    if isinstance(texts, str):
        texts = [texts]
        single_input = True

    embeddings = get_embedding_model(MODEL_NAME).encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,
    )

    if single_input:
        return embeddings[0].tolist()

    return embeddings.tolist()
