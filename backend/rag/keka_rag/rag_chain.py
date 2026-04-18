import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


@lru_cache(maxsize=1)
def get_llm():
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError("GOOGLE_API_KEY not set")

    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0.3,
        google_api_key=api_key
    )


def get_rag_chain(retriever):

    prompt = ChatPromptTemplate.from_template("""
You are an expert assistant for Keka HR platform.

Rules:
- Answer ONLY using the provided context
- If answer is not in context, say "I don't know"
- Be clear and structured
- Use bullet points if needed

Context:
{context}

Question:
{question}

Answer:
""")

    llm = get_llm()

    def format_docs(docs):
        return "\n\n".join(
            f"[{d.metadata.get('file_name')}]\n{d.page_content}"
            for d in docs
        )

    def run(query: str, debug: bool = False):
        docs = retriever.invoke(query)

        if debug:
            print("\n🔍 Retrieved docs:")
            for d in docs:
                print(d.metadata)

        context = format_docs(docs)

        response = llm.invoke(
            prompt.format(
                context=context,
                question=query
            )
        )

        return response.content

    return run
