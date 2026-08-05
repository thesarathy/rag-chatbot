"""
llm.py
Handles communication with the Groq API (fast inference for open models,
e.g. Llama 3.3), including prompt construction that grounds answers
strictly in retrieved document context.
"""

import os
from groq import Groq
from dotenv import load_dotenv
from typing import List, Dict

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = """You are a document Q&A assistant. Answer the user's question
using ONLY the information provided in the context below. Do not use any
outside knowledge, even if you know the answer from elsewhere.

If the context does not contain enough information to answer the question,
say clearly: "I don't have enough information in the provided documents to
answer that." Do not guess or make up an answer.

When you use information from the context, mention which source it came from
if possible.
"""


def build_context(retrieved_chunks: List[Dict]) -> str:
    """
    Format retrieved chunks into a single context string for the prompt,
    with source/page labels so the model can reference them.
    """
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        context_parts.append(
            f"[Source {i}: {chunk['source']}, page {chunk['page']}]\n{chunk['text']}"
        )
    return "\n\n".join(context_parts)


def ask_llm(
    question: str,
    retrieved_chunks: List[Dict],
    model_name: str = "llama-3.3-70b-versatile",
    max_tokens: int = 1000
) -> str:
    """
    Send the question + retrieved context to Groq and return the answer.

    Args:
        question: The user's question.
        retrieved_chunks: Output of retriever.similarity_search().
        model_name: Groq model identifier.
        max_tokens: Max tokens in the response.

    Returns:
        The model's answer as a string.
    """
    context = build_context(retrieved_chunks)

    user_message = f"""Context:
{context}

Question: {question}

Answer the question using only the context above."""

    try:
        response = client.chat.completions.create(
            model=model_name,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ]
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"Error contacting Groq API: {str(e)}"