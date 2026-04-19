from rag.vector_store import VectorStore
from rag.scraper import scrap_website
from config.paths import HEXNODE_VECTOR_INDEX


async def initialize_vector_store(index_path=HEXNODE_VECTOR_INDEX):
    store = VectorStore()

    # Try loading an existing index first.
    if store.load(index_path):
        print("Loaded existing vector store")
        print("Vectors:", store.index.ntotal)
        return store
    
    print("No index found. Creating new embeddings...")

    await scrap_website(store)

    print("New vector store created")
    return store
