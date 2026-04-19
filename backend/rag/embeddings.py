import logging

from sentence_transformers import SentenceTransformer


logger = logging.getLogger(__name__)

MODEL_NAME = "all-MiniLM-L6-v2"
logger.info("Loading embedding model: %s", MODEL_NAME)
_model = SentenceTransformer(MODEL_NAME)


def get_embedding(texts):
    """
    Supports both a single string and a list of strings.
    """
    single_input = False

    if isinstance(texts, str):
        texts = [texts]
        single_input = True

    embeddings = _model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    if single_input:
        return embeddings[0].tolist()

    return embeddings.tolist()
