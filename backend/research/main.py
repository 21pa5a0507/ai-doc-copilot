from research.qdrant_expo import ingest_to_qdrant
from rag.embeddings import get_embedding
from rag.gemini_models import get_genai_client
from qdrant_client import models

COLLECTION_NAME = "docs"


def search_docs(client, query: str, domain):
    print(f"Searching for '{query}' in domain '{domain}'...")
    query_embedding = get_embedding(query)
    return client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=5,
        query_filter=models.Filter(
        must=[
            models.FieldCondition(
                key="domain",
                match=models.MatchValue(
                    value=domain,
                ),
            )
        ]
        ),
        with_payload=True,
    )


def format_search_results(results):
    lines = []
    for point in results.points:
        payload = point.payload or {}
        text = payload.get("text", "")
        source = payload.get("file_name") or payload.get("url") or payload.get("source", "")
        domain = payload.get("domain", "")
        lines.append(f"[{domain}] {source}\n{text}")
    return "\n\n".join(lines)




def main():
    llm = get_genai_client()
    client = ingest_to_qdrant(collection_name=COLLECTION_NAME)

   

   

    while True:
        user_query = input("Enter your search query (or 'exit' to quit): ")
        if user_query.lower() == "exit":
            break
        route_prompt = f"""
You are a domain router for a document search system.

Choose exactly one domain for the user query.

Domains:
- hexnode: IT asset management, MDM, device enrollment, Windows, macOS, Android, iOS, policies, compliance, Hexnode.
- hr: human resources, employee onboarding, benefits, payroll, leave, attendance, employee management.

User query:
{user_query}
Return only one word: hexnode or hr.
"""
        route_response = llm.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=route_prompt,
        )
        domain = route_response.text.strip()
        results = search_docs(client, user_query, domain)
        search_context = format_search_results(results)
        prompt = f"""
You are a helpful assistant.
Answer the user's query using only the relevant documents below.

Relevant documents:
{search_context}

User query:
{user_query}
"""
       
        llm_response = llm.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt,
        )

        print("\nAnswer:")
        print(llm_response.text)
        print("\nSearch results:")
        print(search_context)


if __name__ == "__main__":
    main()
