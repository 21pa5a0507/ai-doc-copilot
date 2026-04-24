import logging
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from config.paths import RERANKER_MODEL_DIR


logger = logging.getLogger(__name__)

MODEL_ONNX_FILE = "model.onnx"


def _resolve_model_dir() -> Path:
    override = os.getenv("ONNX_RERANKER_MODEL_DIR")
    return Path(override) if override else RERANKER_MODEL_DIR


def _resolve_model_file(model_dir: Path) -> Path:
    for candidate in (model_dir / MODEL_ONNX_FILE, model_dir / "onnx" / MODEL_ONNX_FILE):
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"ONNX reranker file not found in {model_dir}. "
        f"Expected {MODEL_ONNX_FILE} or onnx/{MODEL_ONNX_FILE}."
    )


def _to_scores(logits: np.ndarray) -> np.ndarray:
    if logits.ndim == 1:
        return 1.0 / (1.0 + np.exp(-logits))

    if logits.shape[1] == 1:
        return 1.0 / (1.0 + np.exp(-logits[:, 0]))

    shifted = logits - np.max(logits, axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities[:, -1]


class OnnxReranker:
    def __init__(self):
        import onnxruntime as ort

        self.model_dir = _resolve_model_dir()
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

    def predict(self, pairs):
        left_texts = [pair[0] for pair in pairs]
        right_texts = [pair[1] for pair in pairs]
        encoded = self.tokenizer(
            left_texts,
            right_texts,
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
        logits = np.asarray(outputs[0], dtype=np.float32)
        return _to_scores(logits)


@lru_cache(maxsize=1)
def get_reranker_model() -> OnnxReranker:
    model = OnnxReranker()
    logger.info(
        "Loaded ONNX reranker model from %s",
        model.model_dir,
    )
    return model
