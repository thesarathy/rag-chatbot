"""
llm.py
Handles communication with the Anthropic Claude API, including prompt
construction that grounds answers strictly in retrieved document context.
"""

import os
from anthropic import Anthropic
from dotenv import load_dotenv
from typing import List, Dict

load_dotenv()

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

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
    with source/page labels so Claude can reference them.
    """
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        context_parts.append(
            f"[Source {i}: {chunk['source']}, page {chunk['page']}]\n{chunk['text']}"
        )
    return "\n\n".join(context_parts)


def ask_claude(
    question: str,
    retrieved_chunks: List[Dict],
    model: str = "claude-sonnet-4-5",
    max_tokens: int = 1000
) -> str:
    """
    Send the question + retrieved context to Claude and return the answer.

    Args:
        question: The user's question.
        retrieved_chunks: Output of retriever.similarity_search().
        model: Claude model identifier.
        max_tokens: Max tokens in the response.

    Returns:
        Claude's answer as a string.
    """
    context = build_context(retrieved_chunks)

    user_message = f"""Context:
{context}

Question: {question}

Answer the question using only the context above."""

    try:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": user_message}
            ]
        )
        return response.content[0].text

    except Exception as e:
        return f"Error contacting Claude API: {str(e)}"