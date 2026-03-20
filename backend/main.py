import asyncio
import sys

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from fastapi import FastAPI
from rag.answer_generator import generate_answer
from rag.embendings import get_embending as embed_text
from rag.vector_store import VectorStore
from rag.scraper import scrap_website
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel

class Query(BaseModel):
    question: str

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


vector_store = VectorStore()

# Run scraper once when server starts
@app.on_event("startup")
async def startup_event():
    await scrap_website(vector_store)
    print("Vectors stored:", vector_store.index.ntotal)


@app.post("/ask")
def ask(query: Query):
    question = query.question

    query_embedding = embed_text(question)

    chunks = vector_store.search(query_embedding, question)

    answer = generate_answer(question, chunks)

    return {
        "question": question,
        "chunks": chunks,
        "answer": answer
    }