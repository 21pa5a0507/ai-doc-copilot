import os
from rag.vector_store import VectorStore
from rag.scraper import scrap_website

async def initialize_vector_store(json_cache="vector_store.index"):
    store = VectorStore()

    # ✅ Try loading existing index
    if store.load(json_cache):
        print("✅ Loaded existing vector store")
        print("Vectors:", store.index.ntotal)
        return store
    
    print("⚡ No index found. Creating new embeddings...")

    await scrap_website(store)

    print("✅ New vector store created")
    return store