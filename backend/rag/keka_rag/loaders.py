import logging

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from config.paths import KEKA_SOURCE_PDF_DIR


logger = logging.getLogger(__name__)

PDF_FOLDER = KEKA_SOURCE_PDF_DIR


def load_pdfs():
    documents = []

    if not PDF_FOLDER.exists():
        raise FileNotFoundError(f"Folder not found: {PDF_FOLDER}")

    pdf_files = list(PDF_FOLDER.glob("*.pdf"))

    if not pdf_files:
        raise ValueError("No PDF files found in keka_data folder")

    for file in pdf_files:
        logger.info("Loading Keka PDF %s", file.name)

        loader = PyPDFLoader(str(file))
        pages = loader.load()

        for p in pages:
            documents.append(
                Document(
                    page_content=p.page_content,
                    metadata={
                        "source": "keka",
                        "file_name": file.name
                    }
                )
            )

    logger.info("Loaded %s Keka PDF pages", len(documents))
    return documents
