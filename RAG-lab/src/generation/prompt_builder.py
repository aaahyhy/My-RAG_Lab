def build_prompt(query, docs):

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
You are a helpful assistant.

Answer the question using ONLY the context.

Context:
{context}

Question:
{query}
"""

    return prompt