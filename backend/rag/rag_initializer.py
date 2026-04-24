import logging

from rag.vector_store import VectorStore
from config.paths import (
    HEXNODE_EMB_INDEX,
    HEXNODE_META_CACHE,
    LEGACY_HEXNODE_VECTOR_INDEX,
    LEGACY_HEXNODE_VECTOR_META,
)


logger = logging.getLogger(__name__)


async def initialize_vector_store(index_path=HEXNODE_EMB_INDEX, meta_path=HEXNODE_META_CACHE):
    store = VectorStore()

    # Try loading an existing index first.
    if store.load(index_path, meta_path=meta_path):
        logger.info("Loaded existing Hexnode vector store with %s vectors", store.index.ntotal)
        return store

    if store.load(LEGACY_HEXNODE_VECTOR_INDEX, meta_path=LEGACY_HEXNODE_VECTOR_META):
        logger.info("Loaded legacy Hexnode vector store with %s vectors", store.index.ntotal)
        return store
    
    logger.info("No Hexnode index found. Building a new vector store")
    from rag.scraper import scrap_website
    await scrap_website(store)

    logger.info("Created new Hexnode vector store")
    return store
