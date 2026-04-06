from typing import List
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder


# ---------------------------
# Reranker
# ---------------------------
class Reranker:
    def __init__(self):
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def rerank(self, query: str, docs: List[Document], top_k: int = 5):
        pairs = [(query, doc.page_content) for doc in docs]
        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(docs, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [doc for doc, _ in ranked[:top_k]]


# ---------------------------
# Keka Retriever Class (Hybrid + Rerank)
# ---------------------------
class KekaRetriever:
    def __init__(self, vectorstore, docs: List[Document]):
        self.vectorstore = vectorstore
        self.docs = docs
        self.bm25 = get_bm25_retriever(docs)
        self.reranker = Reranker()

    def invoke(self, query: str):
        # 1. Get results from both retrievers
        bm25_docs = self.bm25.get_relevant_documents(query)
        vector_docs = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 10,
                "fetch_k": 30,
                "lambda_mult": 0.7
            }
        ).get_relevant_documents(query)

        # 2. Combine and deduplicate
        combined_docs = bm25_docs + vector_docs
        retrieved_docs = deduplicate(combined_docs)

        # 3. Rerank
        final_docs = self.reranker.rerank(query, retrieved_docs, top_k=5)

        return final_docs


# ---------------------------
# BM25 Retriever (Keyword)
# ---------------------------
def get_bm25_retriever(docs: List[Document]):
    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = 10   # higher for reranking
    return bm25


# ---------------------------
# Deduplicate Docs
# ---------------------------
def deduplicate(docs: List[Document]):
    seen = set()
    unique_docs = []

    for doc in docs:
        content = doc.page_content.strip()
        if content not in seen:
            seen.add(content)
            unique_docs.append(doc)

    return unique_docs


# ---------------------------
# Get Retriever (returns retriever object)
# ---------------------------
def get_retriever(vectorstore, docs: List[Document]):
    return KekaRetriever(vectorstore, docs)