from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(documents):
    if not documents:
        raise ValueError("No documents to split")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks")
    return chunks
