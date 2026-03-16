import faiss
import numpy as np
from rank_bm25 import BM25Okapi
import os
import re


class VectorStore:

    def __init__(self, dimension=384):

        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.text_chunks = []
        self.tokenized_chunks = []

        self.bm25 = None

    def update_bm25(self):
        self.bm25 = BM25Okapi(self.tokenized_chunks)
    def add(self, embedding, text):

        if len(embedding) != self.dimension:
            raise ValueError(
                f"Embedding length {len(embedding)} does not match expected dimension {self.dimension}."
            )

        vector = np.array(embedding, dtype="float32").reshape(1, -1)

        self.index.add(vector)

        self.text_chunks.append(text)

        # prepare for BM25
        tokens = re.findall(r"\w+", chunk["content"].lower())
        self.tokenized_chunks.append(tokens)
        faiss.write_index(self.index, "vector_store.index")

    def vector_search(self, embedding, k=5):

        print(f"✅ Vectors stored: {self.index.ntotal}",self.index.ntotal)

        if self.index.ntotal == 0:
            return []

        vector = np.array(embedding, dtype="float32").reshape(1, -1)

        distances, indices = self.index.search(vector, k)

        results = []

        for i in indices[0]:
            if i != -1 and i < len(self.text_chunks):
                results.append(self.text_chunks[i])

        return results

    def keyword_search(self, query, k=5):

        if not self.bm25:
            return []

        tokenized_query = re.findall(r"\w+", query.lower())

        scores = self.bm25.get_scores(tokenized_query)

        ranked = sorted(
            zip(self.text_chunks, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [chunk for chunk, _ in ranked[:k]]

    def search(self, embedding, query, k=5):

        print("🔍 Performing combined search...")
        vector_results = self.vector_search(embedding, k)

        keyword_results = self.keyword_search(query, k)

        # combine both results
        combined = vector_results + keyword_results

        # remove duplicates
        seen = set()
        unique_results = []
        for chunk in combined:
            content = chunk["content"]
            if content not in seen:
                unique_results.append(chunk)
                seen.add(content)

        return unique_results[:k]