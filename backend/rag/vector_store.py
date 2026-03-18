import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import re


# -------------------------------
# Reranker
# -------------------------------
class Reranker:
    def __init__(self):
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def rerank(self, query, chunks, top_k=5):
        if not chunks:
            return []

        pairs = [
            (query, f"{chunk['title']} {chunk['content']}")
            for chunk in chunks
        ]

        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(chunks, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [chunk for chunk, _ in ranked[:top_k]]


# -------------------------------
# Vector Store
# -------------------------------
class VectorStore:

    def __init__(self, dimension=384):
        self.dimension = dimension

        # Cosine similarity
        self.index = faiss.IndexFlatIP(dimension)

        self.text_chunks = []
        self.tokenized_chunks = []

        self.bm25 = None
        self.reranker = Reranker()

        # simple stopwords
        self.stopwords = {"the", "is", "and", "in", "to", "of", "a", "for"}

    # -------------------------------
    # Utils
    # -------------------------------
    def normalize(self, vec):
        vec = np.array(vec, dtype="float32")
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-10)

    def tokenize(self, text):
        words = re.findall(r"\w+", text.lower())
        return [w for w in words if w not in self.stopwords]

    def normalize_scores(self, results):
        if not results:
            return []

        scores = [score for _, score in results]
        min_s, max_s = min(scores), max(scores)

        return [
            (chunk, (score - min_s) / (max_s - min_s + 1e-6))
            for chunk, score in results
        ]

    # -------------------------------
    # Add Data
    # -------------------------------
    def add(self, embedding, chunk):
        """
        chunk = {
            "title": ...,
            "content": ...,
            "url": ...
        }
        """

        if len(embedding) != self.dimension:
            raise ValueError("Embedding dimension mismatch")

        vector = self.normalize(embedding).reshape(1, -1)
        self.index.add(vector)

        self.text_chunks.append(chunk)

        tokens = self.tokenize(chunk["content"])
        self.tokenized_chunks.append(tokens)

    def build_bm25(self):
        self.bm25 = BM25Okapi(self.tokenized_chunks)

    # -------------------------------
    # Searches
    # -------------------------------
    def vector_search(self, embedding, k=15):
        if self.index.ntotal == 0:
            return []

        vector = self.normalize(embedding).reshape(1, -1)
        scores, indices = self.index.search(vector, k)

        results = []
        for score, i in zip(scores[0], indices[0]):
            if i != -1 and score > 0.2:  # threshold filter
                results.append((self.text_chunks[i], float(score)))

        return results

    def keyword_search(self, query, k=15):
        if not self.bm25:
            return []

        tokenized_query = self.tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        ranked = sorted(
            zip(self.text_chunks, scores),
            key=lambda x: x[1],
            reverse=True
        )

        # filter low scores
        return [
            (chunk, float(score))
            for chunk, score in ranked[:k]
            if score > 0
        ]

    # -------------------------------
    # Hybrid + Rerank
    # -------------------------------
    def search(self, embedding, query, k=5):

        print("🔍 Hybrid search running...")

        # Step 1: Retrieve candidates
        vector_results = self.vector_search(embedding, k=15)
        keyword_results = self.keyword_search(query, k=15)

        # Step 2: Normalize scores
        vector_results = self.normalize_scores(vector_results)
        keyword_results = self.normalize_scores(keyword_results)

        # Step 3: Fusion
        scores_dict = {}

        for chunk, score in vector_results:
            key = id(chunk)
            scores_dict[key] = scores_dict.get(key, 0) + 0.7 * score

        for chunk, score in keyword_results:
            key = id(chunk)
            scores_dict[key] = scores_dict.get(key, 0) + 0.3 * score

        # Step 4: Map back
        id_to_chunk = {id(c): c for c in self.text_chunks}

        ranked = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)

        # Debug (optional)
        print("\nTop hybrid candidates:")
        for key, score in ranked[:5]:
            print(score, id_to_chunk[key]["content"][:80])

        # Step 5: Select top candidates
        top_candidates = [
            id_to_chunk[key]
            for key, _ in ranked[:15]
        ]

        # Step 6: Rerank
        reranked = self.reranker.rerank(query, top_candidates, top_k=k)

        return reranked