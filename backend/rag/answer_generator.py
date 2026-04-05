from google import genai
import os
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY1"))

MODEL_NAME = "gemini-2.5-flash-lite"
def test_model():
    print([m.name for m in client.models.list()])

def rewrite_query(user_query):
    prompt = f"""
    Rewrite the query using FastAPI terms like HTTP methods, path operations, request body, query parameters, and dependencies.

    Query: {user_query}
    Rewritten:
    """
    question = model.generate_content(prompt)

    print(f"🔄 Rewritten query: {question.text.strip()}")

    return question.text.strip()

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

    response =  client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    return response.text
def get_gemini_models():
    """
    Configures the API and returns a list of available models 
    supporting content generation.
    """
    
    # Filter for models that support generating content
    models = [m.name for m in genai.list_models() 
              if 'generateContent' in m.supported_generation_methods]
    return models

if __name__ == "__main__":
    # It is best practice to store your API key in an environment variable
    # Replace 'GEMINI_API_KEY' with your actual key or environment variable name
    # available_models = get_gemini_models()
    test_model()