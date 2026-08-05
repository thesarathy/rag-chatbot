"""
chat.py
Orchestrates the full RAG pipeline: retrieval + generation, with
conversation memory across turns.
"""

from typing import List, Dict
from utils.retriever import similarity_search
from utils.llm import ask_llm, build_context
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


class ChatSession:
    """
    Holds conversation state (history) for one chat session and
    orchestrates retrieval + generation for each new turn.
    """

    def __init__(self, vectorstore, top_k: int = 4):
        self.vectorstore = vectorstore
        self.top_k = top_k
        self.history: List[Dict] = []  # list of {"role": "user"/"assistant", "content": str}

    def _rewrite_query(self, question: str) -> str:
        """
        If there's prior conversation history, rewrite the current question
        into a standalone question using that history, so retrieval works
        correctly on follow-ups like "what about the second one?"

        If there's no history yet, return the question unchanged.
        """
        if not self.history:
            return question

        history_text = "\n".join(
            f"{turn['role']}: {turn['content']}" for turn in self.history[-4:]  # last 2 exchanges
        )

        rewrite_prompt = f"""Given this conversation history:
{history_text}

And this follow-up question: "{question}"

Rewrite the follow-up into a standalone question that makes sense without
the conversation history. If it's already standalone, return it unchanged.
Return ONLY the rewritten question, nothing else."""

        try:
            response = _client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                max_tokens=150,
                messages=[{"role": "user", "content": rewrite_prompt}]
            )
            return response.choices[0].message.content.strip()
        except Exception:
            # If rewriting fails for any reason, fall back to the original question
            # rather than breaking the whole chat turn
            return question

    def ask(self, question: str) -> Dict:
        """
        Process one user turn: retrieve relevant chunks, generate an answer,
        update history, and return everything the UI needs to display.

        Returns:
            {
                "answer": str,
                "retrieved_chunks": List[Dict],
                "standalone_question": str
            }
        """
        standalone_question = self._rewrite_query(question)

        retrieved_chunks = similarity_search(
            self.vectorstore, standalone_question, top_k=self.top_k
        )

        answer = ask_llm(standalone_question, retrieved_chunks)

        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": answer})

        return {
            "answer": answer,
            "retrieved_chunks": retrieved_chunks,
            "standalone_question": standalone_question
        }

    def reset(self):
        """Clear conversation history (e.g. when user starts a new chat)."""
        self.history = []