from fastapi import FastAPI
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)


model = genai.GenerativeModel("gemini-2.5-flash")

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Welcome to the AI Doc Copilot API!"}

@app.get("/ask")
def ask_ai(question: str):
    try:
        response =model.generate_content(question)
        return {
            "question": question,
            "answer": response.text
        }
    except Exception as e:
        return {
            "error": str(e)
        }