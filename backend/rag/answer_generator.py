import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-2.5-flash")

def generate_answer(question, chunks):

    context = "\n\n".join([f"Title: {chunk['title']}\nContent: {chunk['content']}" for chunk in chunks])

    prompt = f"""
    You are a strict AI assistant.

    Answer ONLY from the provided context.
    Do NOT use your own knowledge.

    If the answer is not clearly present in the context, say:
    "Answer not found in the provided documentation."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    response = model.generate_content(prompt)

    return response.text