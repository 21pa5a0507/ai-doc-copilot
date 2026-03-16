import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-2.5-flash")

def generate_answer(question, chunks):

    context = "\n\n".join(chunks)

    prompt = f"""
     You are an assistant that answers questions based on the following context:
     {context}
     Question: {question}
     Answer:
     """

    response = model.generate_content(prompt)

    return response.text