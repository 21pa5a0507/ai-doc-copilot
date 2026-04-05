from typing import List
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer


class HFEmbeddings(Embeddings):
    def __init__(self, model_name="BAAI/bge-base-en"):
        print("🚀 Loading embedding model...")
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        texts = [f"passage: {t}" for t in texts]

        return self.model.encode(
            texts,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).tolist()

    def embed_query(self, text: str) -> List[float]:
        text = f"query: {text}"

        return self.model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True
        )[0].tolist()


# singleton
_embedding_instance = None


def get_embeddings():
    global _embedding_instance
    if _embedding_instance is None:
        _embedding_instance = HFEmbeddings()
    return _embedding_instance