import logging
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from config.paths import HEXNODE_EMBEDDING_MODEL_DIR, KEKA_EMBEDDING_MODEL_DIR


logger = logging.getLogger(__name__)

MODEL_ONNX_FILE = "model.onnx"


def _resolve_model_dir(model_name: str) -> Path:
    if model_name == "all-MiniLM-L6-v2":
        override = os.getenv("ONNX_MODEL_DIR_ALL_MINILM_L6_V2")
        return Path(override) if override else HEXNODE_EMBEDDING_MODEL_DIR

    if model_name == "all-MiniLM-L6-v2-keka":
        override = os.getenv("ONNX_MODEL_DIR_ALL_MINILM_L6_V2_KEKA")
        return Path(override) if override else KEKA_EMBEDDING_MODEL_DIR

    raise ValueError(f"Unsupported embedding model: {model_name}")


def _resolve_model_file(model_dir: Path) -> Path:
    model_file = model_dir / MODEL_ONNX_FILE
    if model_file.exists():
        return model_file

    nested_model_file = model_dir / "onnx" / MODEL_ONNX_FILE
    if nested_model_file.exists():
        return nested_model_file

    raise FileNotFoundError(
        f"ONNX model file not found in {model_dir}. "
        f"Expected {MODEL_ONNX_FILE} or onnx/{MODEL_ONNX_FILE}."
    )


def _mean_pool(token_embeddings: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    mask = attention_mask[..., None].astype(np.float32)
    summed = (token_embeddings * mask).sum(axis=1)
    counts = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)
    return summed / counts


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    return matrix / norms


class OnnxEmbeddingModel:
    def __init__(self, model_name: str):
        import onnxruntime as ort

        self.model_dir = _resolve_model_dir(model_name)
        self.model_file = _resolve_model_file(self.model_dir)
        self.provider = os.getenv("ONNX_PROVIDER", "CPUExecutionProvider")
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_dir),
            local_files_only=True,
        )
        self.session = ort.InferenceSession(
            str(self.model_file),
            providers=[self.provider],
        )
        self.input_names = {input_meta.name for input_meta in self.session.get_inputs()}

    def encode(
        self,
        texts,
        batch_size=32,
        normalize_embeddings=True,
    ):
        batches = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="np",
            )
            feeds = {
                name: encoded[name].astype(np.int64)
                for name in encoded
                if name in self.input_names
            }
            outputs = self.session.run(None, feeds)
            pooled = _mean_pool(outputs[0], encoded["attention_mask"])
            if normalize_embeddings:
                pooled = _normalize_rows(pooled)
            batches.append(pooled.astype(np.float32))

        return np.concatenate(batches, axis=0) if batches else np.empty((0, 0), dtype=np.float32)


@lru_cache(maxsize=8)
def get_embedding_model(model_name: str) -> OnnxEmbeddingModel:
    model = OnnxEmbeddingModel(model_name)
    logger.info(
        "Loaded ONNX embedding model %s from %s",
        model_name,
        model.model_dir,
    )
    return model
