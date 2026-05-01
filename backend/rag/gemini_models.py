import os
import logging
from functools import lru_cache

from google import genai
from google.genai import types


logger = logging.getLogger(__name__)

MODEL_STACK = [
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-3.1-flash-lite-preview",
    "gemini-3.1-flash-preview",
    "gemini-3.1-pro-preview",
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
    return genai.Client(
        api_key=get_google_api_key(),
        http_options=types.HttpOptions(
            retry_options=types.HttpRetryOptions(attempts=0),
        ),
    )


def generate_text_with_fallback(client, prompt_text):
    for model_id in MODEL_STACK:
        try:
            response = client.models.generate_content(
                model=model_id,
                contents=prompt_text,
            )
            text = getattr(response, "text", None)

            if text and text.strip():
                logger.info("Generated response with %s", model_id)
                return text.strip()

            logger.warning("Empty response from %s. Trying next model.", model_id)
        except Exception as exc:
            status_code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
            if status_code is None and getattr(exc, "response", None) is not None:
                status_code = getattr(exc.response, "status_code", None)

            message = str(exc).lower()
            is_retryable = status_code in {408, 429, 500, 502, 503, 504} or any(
                marker in message
                for marker in ("timeout", "timed out", "connection", "service unavailable")
            )

            if not is_retryable:
                logger.exception("%s failed with a non-retryable error", model_id)
                raise

            logger.warning("%s failed with a retryable error: %s", model_id, exc)

    return None
