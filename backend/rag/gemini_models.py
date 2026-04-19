import os
from functools import lru_cache

from google import genai


MODEL_STACK = [
    "gemini-3.1-flash-lite-preview",
    "gemini-3.1-flash-preview",
    "gemini-2.5-flash-lite",
    "gemini-3.1-pro-preview",
    "gemini-2.5-flash",
]

PRIMARY_MODEL = MODEL_STACK[0]


@lru_cache(maxsize=1)
def get_genai_client():
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY1")

    if not api_key:
        raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY1 not set")

    return genai.Client(api_key=api_key)


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

            print(f"Empty response from {model_id}. Trying next model...")
        except Exception as exc:
            print(f"{model_id} failed: {exc}")

    return None
