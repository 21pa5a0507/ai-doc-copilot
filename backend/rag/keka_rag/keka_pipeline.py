from keka_rag.loaders import load_pdfs
from keka_rag.splitter import split_documents
from keka_rag.vector_store import get_vectorstore
from keka_rag.retriever import get_retriever
from keka_rag.rag_chain import get_rag_chain
import pickle
import os

CACHE_FILE = "keka_faiss/keka_docs_cache.pkl"

def load_cached_docs():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"⚠️ Failed to load cached docs: {e}")
    return None

def save_cached_docs(docs):
    try:
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(docs, f)
        print("✅ Cached docs saved to file")
    except Exception as e:
        print(f"⚠️ Failed to save cached docs: {e}")


def main():
    # Load cached docs first
    chunks = load_cached_docs()

    if chunks is None:
        # Cache docs
        docs = load_pdfs()
        chunks = split_documents(docs)
        save_cached_docs(chunks)

    try:
        # Try loading existing FAISS
        vectorstore = get_vectorstore()

    except:
        print("⚠️ No FAISS found. Creating new one...")
        vectorstore = get_vectorstore(chunks)

    retriever = get_retriever(vectorstore, chunks)
    rag = get_rag_chain(retriever)

    while True:
        query = input("\nAsk something (or type 'exit'): ")

        if query.lower() == "exit":
            break

        answer = rag(query, debug=False)
        print("\n💡 Answer:\n", answer)


if __name__ == "__main__":
    main()