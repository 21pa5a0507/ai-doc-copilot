from typing import List
import time

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

from rag.backends.onnx_reranker import get_reranker_model


class Reranker:
    def __init__(self):
        self.model = get_reranker_model()

    def rerank(self, query: str, docs: List[Document], top_k: int = 5):
        pairs = [(query, doc.page_content) for doc in docs]
        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(docs, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        return [doc for doc, _ in ranked[:top_k]]


class KekaRetriever:
    def __init__(self, vectorstore, docs: List[Document]):
        self.vectorstore = vectorstore
        self.docs = docs
        self.bm25 = get_bm25_retriever(docs)
        self.reranker = Reranker()
        self.vector_retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 10,
                "fetch_k": 30,
                "lambda_mult": 0.7,
            },
        )
        self.last_timings = {}

    def invoke(self, query: str):
        total_start = time.perf_counter()

        bm25_start = time.perf_counter()
        bm25_docs = self.bm25.invoke(query)
        bm25_seconds = round(time.perf_counter() - bm25_start, 4)

        vector_start = time.perf_counter()
        vector_docs = self.vector_retriever.invoke(query)
        vector_seconds = round(time.perf_counter() - vector_start, 4)

        combined_docs = bm25_docs + vector_docs
        retrieved_docs = deduplicate(combined_docs)

        rerank_start = time.perf_counter()
        final_docs = self.reranker.rerank(query, retrieved_docs, top_k=5)
        rerank_seconds = round(time.perf_counter() - rerank_start, 4)

        self.last_timings = {
            "bm25_seconds": bm25_seconds,
            "vector_search_seconds": vector_seconds,
            "rerank_seconds": rerank_seconds,
            "retrieval_total_seconds": round(time.perf_counter() - total_start, 4),
        }

        return final_docs


def get_bm25_retriever(docs: List[Document]):
    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = 10
    return bm25


def deduplicate(docs: List[Document]):
    seen = set()
    unique_docs = []

    for doc in docs:
        content = doc.page_content.strip()
        if content not in seen:
            seen.add(content)
            unique_docs.append(doc)

    return unique_docs


def get_retriever(vectorstore, docs: List[Document]):
    return KekaRetriever(vectorstore, docs)
