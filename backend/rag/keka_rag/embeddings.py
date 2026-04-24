from typing import List
from langchain_core.embeddings import Embeddings
from rag.backends.onnx_embeddings import get_embedding_model


class HFEmbeddings(Embeddings):
    def __init__(self, model_name="all-MiniLM-L6-v2-keka"):
        self.model = get_embedding_model(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True
        ).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(
            [text],
            normalize_embeddings=True
        )[0].tolist()


# Singleton instance
_embedding_instance = None


def get_embeddings():
    global _embedding_instance
    if _embedding_instance is None:
        _embedding_instance = HFEmbeddings()
    return _embedding_instance
