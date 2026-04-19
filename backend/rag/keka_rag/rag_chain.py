from functools import lru_cache

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from rag.content import content_to_text
from rag.gemini_models import (
    PRIMARY_MODEL,
    generate_text_with_fallback,
    get_genai_client,
    get_google_api_key,
)

load_dotenv()


@lru_cache(maxsize=1)
def get_llm():
    return ChatGoogleGenerativeAI(
        model=PRIMARY_MODEL,
        temperature=0.3,
        google_api_key=get_google_api_key(),
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
    client = get_genai_client()

    def format_docs(docs):
        return "\n\n".join(
            f"[{d.metadata.get('file_name')}]\n{d.page_content}"
            for d in docs
        )

    def answer_with_context(query: str, context: str):
        rendered_prompt = prompt.format(
            context=context,
            question=query
        )

        try:
            response = llm.invoke(rendered_prompt)
            return content_to_text(response.content)
        except Exception as exc:
            print(f"Keka rag_chain fallback triggered: {exc}")

        answer = generate_text_with_fallback(client, rendered_prompt)
        if answer:
            return answer

        return "I don't know"

    def run(query: str, debug: bool = False):
        docs = retriever.invoke(query)

        if debug:
            print("\nRetrieved docs:")
            for d in docs:
                print(d.metadata)

        context = format_docs(docs)
        return answer_with_context(query, context)

    run.answer_with_context = answer_with_context

    return run
