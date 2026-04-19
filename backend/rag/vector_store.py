import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import os
import pickle
from pathlib import Path

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


class VectorStore:
    def __init__(self, dim=384):
        # FAISS index (cosine similarity using inner product)
        self.index = faiss.IndexFlatIP(dim)

        # Stored data
        self.chunks = []
        self.embeddings = []

        # BM25
        self.tokenized_chunks = []
        self.bm25 = None
        self.reranker = Reranker()

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))

        meta_path = path.with_name(path.name + "_meta.pkl")
        with meta_path.open("wb") as f:
            pickle.dump({
                "text_chunks": self.chunks,
                "tokenized_chunks": self.tokenized_chunks
            }, f)
    
    def load(self, path):
        path = Path(path)
        meta_path = path.with_name(path.name + "_meta.pkl")

        if path.exists():
            self.index = faiss.read_index(str(path))

            with meta_path.open("rb") as f:
                meta = pickle.load(f)
                self.chunks = meta["text_chunks"]
                self.tokenized_chunks = meta["tokenized_chunks"]

            self.build_bm25()

            return True
        else:
            return False

    # -----------------------------
    # ADD DATA
    # -----------------------------
    def add(self, embedding, chunk):
        embedding = np.array(embedding).astype("float32")

        # Normalize for cosine similarity
        embedding = embedding / np.linalg.norm(embedding)

        self.index.add(np.array([embedding]))

        self.embeddings.append(embedding)
        self.chunks.append(chunk)

        # Tokenize for BM25
        tokens = chunk["content"].lower().split()
        self.tokenized_chunks.append(tokens)

    # -----------------------------
    # BUILD BM25 (call AFTER all data added)
    # -----------------------------
    def build_bm25(self):
        if not self.tokenized_chunks:
            print("No chunks for BM25")
            return

        self.bm25 = BM25Okapi(self.tokenized_chunks)
        print(f"BM25 built on {len(self.tokenized_chunks)} chunks")

    # -----------------------------
    # SEARCH (HYBRID)
    # -----------------------------
    def search(self, query_embedding, query, top_k=5):
        query_embedding = np.array(query_embedding).astype("float32")

        # Normalize query embedding
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        # -----------------------------
        # 1. VECTOR SEARCH
        # -----------------------------
        distances, indices = self.index.search(np.array([query_embedding]), top_k * 2)

        vector_results = []
        for score, idx in zip(distances[0], indices[0]):
            if idx != -1:
                vector_results.append({
                    "chunk": self.chunks[idx],
                    "score": float(score),
                    "source": "vector"
                })

        # -----------------------------
        # 2. BM25 SEARCH
        # -----------------------------
        bm25_results = []
        if self.bm25:
            tokenized_query = query.lower().split()
            scores = self.bm25.get_scores(tokenized_query)

            top_indices = np.argsort(scores)[-top_k * 2:][::-1]

            for idx in top_indices:
                bm25_results.append({
                    "chunk": self.chunks[idx],
                    "score": float(scores[idx]),
                    "source": "bm25"
                })

        # -----------------------------
        # 3. NORMALIZATION FUNCTION
        # -----------------------------
        def normalize(results):
            if not results:
                return results

            scores = [r["score"] for r in results]
            min_s, max_s = min(scores), max(scores)

            for r in results:
                if max_s - min_s > 0:
                    r["score"] = (r["score"] - min_s) / (max_s - min_s)
                else:
                    r["score"] = 0.5

            return results

        vector_results = normalize(vector_results)
        bm25_results = normalize(bm25_results)

        # -----------------------------
        # 4. COMBINE
        # -----------------------------
        combined = vector_results + bm25_results

        # -----------------------------
        # 5. SORT
        # -----------------------------
        combined = sorted(combined, key=lambda x: x["score"], reverse=True)

        # -----------------------------
        # 6. DEDUPLICATION
        # -----------------------------
        seen = set()
        final_results = []

        for result in combined:
            content = result["chunk"]["content"]

            if content not in seen:
                seen.add(content)
                final_results.append(result["chunk"])

            if len(final_results) >= top_k:
                break

        reranked = self.reranker.rerank(query, final_results, top_k=top_k)
        print(
            "Hybrid search candidates: "
            f"vector={len(vector_results)}, "
            f"bm25={len(bm25_results)}, "
            f"unique={len(final_results)}, "
            f"reranked={len(reranked)}"
        )
        return reranked
