import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-2.5-flash")

def generate_answer(question, chunks):

    context = "\n\n".join([f"Title: {chunk['title']}\nContent: {chunk['content']}" for chunk in chunks])

    prompt = f"""
    You are a helpful AI assistant.
    Use the provided context to answer the question.
    if the context is insufficient, you can use your general knowledge.
    Context:
    {context}
    Question: {question}
    Answer clearly and concisely.
    """

    response = model.generate_content(prompt)

    return response.text