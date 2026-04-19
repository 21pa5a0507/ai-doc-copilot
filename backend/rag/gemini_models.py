import os
import logging
from functools import lru_cache

from google import genai


logger = logging.getLogger(__name__)

MODEL_STACK = [
    "gemini-2.5-flash-lite",
    "gemini-3.1-flash-lite-preview",
    "gemini-3.1-flash-preview",
    "gemini-3.1-pro-preview",
    "gemini-2.5-flash",
]

PRIMARY_MODEL = MODEL_STACK[0]


def get_google_api_key() -> str:
    api_key = os.getenv("GOOGLE_API_KEY")

    if api_key:
        return api_key

    legacy_api_key = os.getenv("GEMINI_API_KEY1")
    if legacy_api_key:
        logger.warning(
            "GEMINI_API_KEY1 is deprecated. Prefer GOOGLE_API_KEY for Gemini access."
        )
        return legacy_api_key

    raise ValueError("GOOGLE_API_KEY not set")


@lru_cache(maxsize=1)
def get_genai_client():
    return genai.Client(api_key=get_google_api_key())


def generate_text_with_fallback(client, prompt_text):
    for model_id in MODEL_STACK:
        try:
            response = client.models.generate_content(
                model=model_id,
                contents=prompt_text,
            )
            text = getattr(response, "text", None)

            if text and text.strip():
                return text.strip()

            logger.warning("Empty response from %s. Trying next model.", model_id)
        except Exception as exc:
            logger.warning("%s failed: %s", model_id, exc)

    return None
