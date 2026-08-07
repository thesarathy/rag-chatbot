"""
llm.py
Handles communication with the Groq API, including prompt construction
that grounds answers strictly in retrieved document context.
"""

import os
from groq import Groq
from dotenv import load_dotenv
from typing import List, Dict

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = """You are a document Q&A assistant. Answer the user's question
using ONLY the information provided in the context below — never outside
knowledge, even if you know the answer from elsewhere.

Every claim in your answer falls into one of three categories. Decide which
one applies BEFORE you start writing, and commit to that decision — do not
begin an INFERRED attempt and then abandon it partway through:

1. EXPLICIT — directly stated in the context. Answer normally and cite the source.

2. INFERRED — not stated verbatim, but reasonably derivable by combining two
or more facts that ARE present in the context (e.g. applying a general
statement to a specific case it clearly covers). State it plainly, e.g.:
"The guide doesn't give this figure directly for [specific case], but its
general estimate of [X] applies here because [reason]." Never present an
INFERRED claim as EXPLICIT, and never mix an INFERRED attempt with a refusal
in the same answer — pick one.

3. UNSUPPORTED — the context doesn't contain the information and it can't be
reasonably inferred from what IS present. Say clearly: "I don't have enough
information in the provided documents to answer that." Do not guess, and do
not follow this with a partial inferred guess afterward.

If multiple distinct sources are each relevant to the question, list every
relevant one rather than answering with only the single best match. If two
sources describe the same project, merge them into one point instead of
citing the same project twice. When a source's project/section name is known,
name it explicitly (e.g. "Project 15: Agent Orchestration System") instead of
referring to it only as "Source N."
"""


def build_context(retrieved_chunks: List[Dict]) -> str:
    """
    Format retrieved chunks into a single context string for the prompt,
    with source/page/project labels so the model can reference them by name.
    """
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        project_label = chunk.get("project")
        label = f", {project_label}" if project_label else ""
        context_parts.append(
            f"[Source {i}: {chunk['source']}, page {chunk['page']}{label}]\n{chunk['text']}"
        )
    return "\n\n".join(context_parts)


def ask_llm(
    question: str,
    retrieved_chunks: List[Dict],
    model_name: str = "llama-3.3-70b-versatile",
    max_tokens: int = 1000
) -> str:
    context = build_context(retrieved_chunks)

    user_message = f"""Context:
{context}

Question: {question}

Answer the question using only the context above."""

    try:
        response = client.chat.completions.create(
    model=model_name,
    max_tokens=max_tokens,
    temperature=0,
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message}
    ]
)
        return response.choices[0].message.content

    except Exception as e:
        return f"Error contacting Groq API: {str(e)}"