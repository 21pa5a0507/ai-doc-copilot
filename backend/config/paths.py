from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "backend"
SOURCE_DATA_DIR = PROJECT_ROOT / "keka_data"
STORAGE_DIR = PROJECT_ROOT / "storage"
BACKEND_LOG_FILE = STORAGE_DIR / "backend.log"
MODEL_DIR = BACKEND_DIR / "models"

HEXNODE_EMBEDDING_MODEL_DIR = MODEL_DIR / "hexnode_embedding"
KEKA_EMBEDDING_MODEL_DIR = MODEL_DIR / "hexnode_embedding"
RERANKER_MODEL_DIR = MODEL_DIR / "reranker"

HEXNODE_DIR = STORAGE_DIR / "hexnode"
HEXNODE_RAW_DIR = HEXNODE_DIR / "raw"
HEXNODE_CACHE_DIR = HEXNODE_DIR / "cache"
HEXNODE_INDEX_DIR = HEXNODE_DIR / "index"
HEXNODE_RAW_CACHE = HEXNODE_RAW_DIR / "docs_raw.json"
HEXNODE_CHUNK_CACHE = HEXNODE_CACHE_DIR / "docs_chunk.json"
HEXNODE_EMB_INDEX = HEXNODE_INDEX_DIR / "vector_store_emb.index"
HEXNODE_META_CACHE = HEXNODE_INDEX_DIR / "vector_store_meta.pkl"
LEGACY_HEXNODE_DOCS_JSON = HEXNODE_RAW_DIR / "docs.json"
LEGACY_HEXNODE_VECTOR_INDEX = HEXNODE_INDEX_DIR / "vector_store.index"
LEGACY_HEXNODE_VECTOR_META = HEXNODE_INDEX_DIR / "vector_store.index_meta.pkl"

KEKA_DIR = STORAGE_DIR / "keka"
KEKA_CACHE_DIR = KEKA_DIR / "cache"
KEKA_INDEX_DIR = KEKA_DIR / "index"
KEKA_CHUNK_CACHE = KEKA_CACHE_DIR / "docs_chunk.pkl"
KEKA_FAISS_EMB_DIR = KEKA_INDEX_DIR / "faiss_emb"
LEGACY_KEKA_DOCS_CACHE = KEKA_CACHE_DIR / "docs_cache.pkl"
LEGACY_KEKA_FAISS_DIR = KEKA_INDEX_DIR / "faiss"
KEKA_SOURCE_PDF_DIR = SOURCE_DATA_DIR

TMP_DIR = STORAGE_DIR / "tmp"


def ensure_storage_dirs() -> None:
    for path in (
        HEXNODE_RAW_DIR,
        HEXNODE_CACHE_DIR,
        HEXNODE_INDEX_DIR,
        KEKA_CACHE_DIR,
        KEKA_FAISS_EMB_DIR,
        TMP_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
