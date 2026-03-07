import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

def get_embending(text):

    for model in genai.list_models():
        # We filter for models that support 'embedContent'
        if 'embedContent' in model.supported_generation_methods:
            print(f"Model Name: {model.name}")
            print(f"Description: {model.description}\n")
    result = genai.embed_content(model="models/gemini-embedding-001", content=text)

    return result["embedding"]
