import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter


logger = logging.getLogger(__name__)


def split_documents(documents):
    if not documents:
        raise ValueError("No documents to split")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    logger.info("Created %s Keka chunks", len(chunks))
    return chunks
