from keka_rag.loaders import load_pdfs
from keka_rag.splitter import split_documents
from keka_rag.vector_store import get_vectorstore
from keka_rag.retriever import get_retriever
from keka_rag.rag_chain import get_rag_chain


def main():
    try:
        # Try loading existing FAISS
        vectorstore = get_vectorstore()

    except:
        print("⚠️ No FAISS found. Creating new one...")

        docs = load_pdfs()
        chunks = split_documents(docs)

        vectorstore = get_vectorstore(chunks)

    retriever = get_retriever(vectorstore)
    rag = get_rag_chain(retriever)

    while True:
        query = input("\nAsk something (or type 'exit'): ")

        if query.lower() == "exit":
            break

        answer = rag(query, debug=False)
        print("\n💡 Answer:\n", answer)


if __name__ == "__main__":
    main()