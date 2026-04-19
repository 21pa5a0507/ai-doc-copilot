import logging

from rag.vector_store import VectorStore
from rag.scraper import scrap_website
from config.paths import HEXNODE_VECTOR_INDEX


logger = logging.getLogger(__name__)


async def initialize_vector_store(index_path=HEXNODE_VECTOR_INDEX):
    store = VectorStore()

    # Try loading an existing index first.
    if store.load(index_path):
        logger.info("Loaded existing Hexnode vector store with %s vectors", store.index.ntotal)
        return store
    
    logger.info("No Hexnode index found. Building a new vector store")

    await scrap_website(store)

    logger.info("Created new Hexnode vector store")
    return store
