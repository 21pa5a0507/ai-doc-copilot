import os
from dotenv import load_dotenv
from rag.gemini_models import PRIMARY_MODEL, generate_text_with_fallback, get_genai_client

load_dotenv()

client = get_genai_client()

MODEL_NAME = PRIMARY_MODEL


def rewrite_query(user_query):
    prompt = f"""
    Rewrite the user question into a concise search-friendly query.
    Preserve the original meaning.
    Do not add new details.

    Query: {user_query}
    Rewritten:
    """
    try:
        rewritten = generate_text_with_fallback(client, prompt)
        if not rewritten:
            return user_query.strip()
        print(f"Rewritten query: {rewritten}")
        return rewritten
    except Exception as exc:
        print(f"Query rewrite failed: {exc}")
        return user_query.strip()


def generate_answer(question, chunks, mode="answer"):
    """
    Generate an answer using the provided context chunks.
    """

    context = "\n\n".join([f"Title: {chunk['title']}\nContent: {chunk['content']}" for chunk in chunks])

    # Base instructions to ensure consistent formatting across all modes
    base_instructions = (
        "You are an expert Hexnode Documentation Assistant. "
        "Use ONLY the provided context. If the answer isn't there, say 'Information not found.' "
        "Format using Markdown: **bold** for UI elements, `code` for values, and ### for headers."
    )

    if mode == "topics":
        prompt = f"""
        {base_instructions}
        
        ### TASK: CATEGORIZED OVERVIEW
        1. Review the context and identify the main themes (e.g., Enrollment, Security, Network).
        2. Group related features or settings under these themes.
        3. Remove any duplicate information.
        4. Present the information as a clean, high-level summary.

        ### CONTEXT:
        {context}

        ### QUESTION:
        {question}

        ### OUTPUT FORMAT:
        ## [Category Name]
        - **[Feature Name]**: Brief description.
        """

    elif mode == "steps":
        prompt = f"""
        {base_instructions}
        
        ### TASK: PROCEDURAL STEP EXTRACTION
        1. Determine the exact action the user needs to complete.
        2. Convert the provided context into a clear, ordered sequence of steps.
        3. Use a numbered list and keep the sequence chronological.
        4. For UI navigation, use the format: Navigate to **Tab** > **Setting** > **Sub-setting**.
        5. If the context does not include an explicit procedure, infer the closest actionable guidance and answer directly.
        6. Do NOT say the tool failed, or that steps were not found.
        7. Do NOT mention the documentation, the context, or the tool in the answer.

        ### CONTEXT:
        {context}

        ### QUESTION:
        {question}

        ### OUTPUT FORMAT:
        1. **Step 1**: First action.
        2. **Step 2**: Next action.
        3. **Step 3**: Next action.
        (If a procedure cannot be extracted, provide a short direct answer instead.)
        """

    else: # Default "answer" mode
        prompt = f"""
        {base_instructions}
        
        ### TASK: DIRECT Q&A
        1. Find the specific answer to the user's question.
        2. Provide a concise but complete response.
        3. Use bullet points only if listing 3 or more related items.
        4. Do not include introductory fluff like "Based on the docs..."

        ### CONTEXT:
        {context}

        ### QUESTION:
        {question}

        ### FINAL ANSWER:
        """

    try:
        answer = generate_text_with_fallback(client, prompt)
        if answer:
            return answer
    except Exception as exc:
        print(f"Answer generation failed: {exc}")

    if not chunks:
        return "Information not found."

    if mode == "topics":
        return chunks[0].get("content", "").strip() or "Information not found."

    return chunks[0].get("content", "").strip() or "Information not found."


def get_gemini_models():
    """
    Returns model names from the configured Gemini client.
    Falls back gracefully if generation-method metadata is unavailable.
    """
    available_models = []

    try:
        models = client.models.list()

        for model in models:
            name = getattr(model, "name", None)
            if not name:
                continue

            methods = getattr(model, "supported_generation_methods", [])

            # Some SDK/model responses may not expose supported_generation_methods.
            # In that case, still include the model name rather than failing silently.
            if not methods or "generateContent" in methods:
                available_models.append(name)

        return sorted(set(available_models))

    except Exception as e:
        return [f"Error fetching models: {e}"]


if __name__ == "__main__":
    available_models = get_gemini_models()
    print("Available Gemini models that support content generation:")
    for model in available_models:
        print(f"- {model}")
