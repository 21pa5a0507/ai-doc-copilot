import asyncio
import logging
import uuid
from pathlib import Path
from typing import List

from qdrant_client import QdrantClient, models

from config.paths import (
    HEXNODE_CHUNK_CACHE,
    HEXNODE_RAW_CACHE,
    LEGACY_HEXNODE_DOCS_JSON,
)
from rag.embeddings import get_embedding
from rag.keka_rag.embeddings import get_embeddings
from rag.keka_rag.service import load_or_create_docs
from rag.scraper import (
    chunking_docs,
    get_all_urls,
    crawl_with_depth,
    load_json_cache,
    save_json_cache,
)


logger = logging.getLogger(__name__)

VECTOR_SIZE = 384
DEFAULT_BATCH_SIZE = 128
POINT_ID_NAMESPACE = uuid.UUID("7a54c37d-47b0-4f7e-8f5c-74256a31f9ee")
RESEARCH_DIR = Path(__file__).resolve().parent
DEFAULT_QDRANT_PATH = RESEARCH_DIR / "qdrant_storage"


def create_get_qdrant_client(path: str):
    return QdrantClient(path=path)


def ensure_collection(client, collection_name: str, vector_size: int = VECTOR_SIZE):
    if client.collection_exists(collection_name):
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE,
        ),
    )


def stable_point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(POINT_ID_NAMESPACE, chunk_id))


def build_hr_points() -> List[models.PointStruct]:
    docs = load_or_create_docs()
    texts = [doc.page_content for doc in docs]
    embeddings = get_embeddings().embed_documents(texts)

    points = []
    for index, (doc, vector) in enumerate(zip(docs, embeddings)):
        chunk_id = f"hr-{index}"
        payload = {
            **doc.metadata,
            "text": doc.page_content,
            "domain": "hr",
            "chunk_id": chunk_id,
        }
        points.append(
            models.PointStruct(
                id=stable_point_id(chunk_id),
                vector=vector,
                payload=payload,
            )
        )

    logger.info("Built %s HR Qdrant points", len(points))
    return points


def load_or_create_hexnode_chunks():
    chunked_docs = load_json_cache(HEXNODE_CHUNK_CACHE)
    if chunked_docs is not None:
        return chunked_docs

    docs = load_json_cache(HEXNODE_RAW_CACHE, LEGACY_HEXNODE_DOCS_JSON)
    if docs is None:
        docs = asyncio.run(crawl_with_depth(get_all_urls()))
        save_json_cache(HEXNODE_RAW_CACHE, docs)

    chunked_docs = chunking_docs(docs)
    save_json_cache(HEXNODE_CHUNK_CACHE, chunked_docs)
    return chunked_docs


def build_hexnode_points() -> List[models.PointStruct]:
    docs = load_or_create_hexnode_chunks()
    texts = [doc["content"] for doc in docs]
    embeddings = get_embedding(texts)

    points = []
    for index, (doc, vector) in enumerate(zip(docs, embeddings)):
        chunk_id = f"hexnode-{index}"
        payload = {
            "text": doc["content"],
            "title": doc.get("title", ""),
            "url": doc.get("url", ""),
            "domain": "hexnode",
            "chunk_id": chunk_id,
        }
        points.append(
            models.PointStruct(
                id=stable_point_id(chunk_id),
                vector=vector,
                payload=payload,
            )
        )

    logger.info("Built %s Hexnode Qdrant points", len(points))
    return points


def ingest_to_qdrant(
    path: str = str(DEFAULT_QDRANT_PATH),
    collection_name: str = "docs",
    batch_size: int = DEFAULT_BATCH_SIZE,
):
    client = create_get_qdrant_client(path)
    ensure_collection(client, collection_name)

    if client.count(collection_name=collection_name).count > 0:
        logger.info("Qdrant collection %s already has data, skipping ingestion", collection_name)
        return client
    points = build_hr_points()
    points.extend(build_hexnode_points())

    for start in range(0, len(points), batch_size):
        batch = points[start:start + batch_size]
        client.upsert(collection_name=collection_name, points=batch, wait=True)

    logger.info("Upserted %s points into Qdrant collection %s", len(points), collection_name)
    return client


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = ingest_to_qdrant()
    print(result)
