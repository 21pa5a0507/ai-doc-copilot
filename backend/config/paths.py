from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "backend"
SOURCE_DATA_DIR = PROJECT_ROOT / "keka_data"
STORAGE_DIR = PROJECT_ROOT / "storage"
BACKEND_LOG_FILE = STORAGE_DIR / "backend.log"

HEXNODE_DIR = STORAGE_DIR / "hexnode"
HEXNODE_RAW_DIR = HEXNODE_DIR / "raw"
HEXNODE_INDEX_DIR = HEXNODE_DIR / "index"
HEXNODE_DOCS_JSON = HEXNODE_RAW_DIR / "docs.json"
HEXNODE_VECTOR_INDEX = HEXNODE_INDEX_DIR / "vector_store.index"
HEXNODE_VECTOR_META = HEXNODE_INDEX_DIR / "vector_store.index_meta.pkl"

KEKA_DIR = STORAGE_DIR / "keka"
KEKA_CACHE_DIR = KEKA_DIR / "cache"
KEKA_INDEX_DIR = KEKA_DIR / "index"
KEKA_FAISS_DIR = KEKA_INDEX_DIR / "faiss"
KEKA_DOCS_CACHE = KEKA_CACHE_DIR / "docs_cache.pkl"
KEKA_SOURCE_PDF_DIR = SOURCE_DATA_DIR

TMP_DIR = STORAGE_DIR / "tmp"


def ensure_storage_dirs() -> None:
    for path in (
        HEXNODE_RAW_DIR,
        HEXNODE_INDEX_DIR,
        KEKA_CACHE_DIR,
        KEKA_FAISS_DIR,
        TMP_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
