import faiss
import numpy as np
import logging
import os
import pickle
import time
from pathlib import Path

from rank_bm25 import BM25Okapi
from rag.backends.onnx_reranker import get_reranker_model


logger = logging.getLogger(__name__)


class Reranker:
    def __init__(self):
        self.model = get_reranker_model()

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
            reverse=True,
        )

        return [chunk for chunk, _ in ranked[:top_k]]


class VectorStore:
    def __init__(self, dim=384):
        self.index = faiss.IndexFlatIP(dim)
        self.chunks = []
        self.embeddings = []
        self.tokenized_chunks = []
        self.bm25 = None
        self.reranker = Reranker()
        self.last_timings = {}

    def save(self, path, meta_path=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))

        meta_path = Path(meta_path) if meta_path else path.with_name(path.name + "_meta.pkl")
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with meta_path.open("wb") as f:
            pickle.dump(
                {
                    "text_chunks": self.chunks,
                    "tokenized_chunks": self.tokenized_chunks,
                },
                f,
            )

    def load(self, path, meta_path=None):
        path = Path(path)
        meta_path = Path(meta_path) if meta_path else path.with_name(path.name + "_meta.pkl")

        if path.exists() and meta_path.exists():
            self.index = faiss.read_index(str(path))

            with meta_path.open("rb") as f:
                meta = pickle.load(f)
                self.chunks = meta["text_chunks"]
                self.tokenized_chunks = meta["tokenized_chunks"]

            self.build_bm25()

            return True

        return False

    def add(self, embedding, chunk):
        embedding = np.array(embedding).astype("float32")

        embedding = embedding / np.linalg.norm(embedding)

        self.index.add(np.array([embedding]))

        self.embeddings.append(embedding)
        self.chunks.append(chunk)

        tokens = chunk["content"].lower().split()
        self.tokenized_chunks.append(tokens)

    def build_bm25(self):
        if not self.tokenized_chunks:
            logger.info("Skipping BM25 build because no chunks were loaded")
            return

        self.bm25 = BM25Okapi(self.tokenized_chunks)
        logger.info("BM25 built on %s chunks", len(self.tokenized_chunks))

    def search(self, query_embedding, query, top_k=5):
        total_start = time.perf_counter()
        timings = {}

        vector_start = time.perf_counter()
        query_embedding = np.array(query_embedding).astype("float32")

        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        distances, indices = self.index.search(np.array([query_embedding]), top_k * 2)
        timings["vector_search_seconds"] = round(time.perf_counter() - vector_start, 4)

        vector_results = []
        for score, idx in zip(distances[0], indices[0]):
            if idx != -1:
                vector_results.append({
                    "chunk": self.chunks[idx],
                    "score": float(score),
                    "source": "vector"
                })

        bm25_results = []
        if self.bm25:
            bm25_start = time.perf_counter()
            tokenized_query = query.lower().split()
            scores = self.bm25.get_scores(tokenized_query)

            top_indices = np.argsort(scores)[-top_k * 2:][::-1]

            for idx in top_indices:
                bm25_results.append({
                    "chunk": self.chunks[idx],
                    "score": float(scores[idx]),
                    "source": "bm25"
                })
            timings["bm25_seconds"] = round(time.perf_counter() - bm25_start, 4)
        else:
            timings["bm25_seconds"] = 0.0

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

        combined = vector_results + bm25_results

        combined = sorted(combined, key=lambda x: x["score"], reverse=True)

        seen = set()
        final_results = []

        for result in combined:
            content = result["chunk"]["content"]

            if content not in seen:
                seen.add(content)
                final_results.append(result["chunk"])

            if len(final_results) >= top_k:
                break

        rerank_start = time.perf_counter()
        reranked = self.reranker.rerank(query, final_results, top_k=top_k)
        timings["rerank_seconds"] = round(time.perf_counter() - rerank_start, 4)
        timings["retrieval_total_seconds"] = round(time.perf_counter() - total_start, 4)
        self.last_timings = timings

        logger.info(
            "Hybrid search candidates: vector=%s, bm25=%s, unique=%s, reranked=%s",
            len(vector_results),
            len(bm25_results),
            len(final_results),
            len(reranked),
        )
        return reranked
