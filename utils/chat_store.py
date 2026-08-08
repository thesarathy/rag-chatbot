"""
chat_store.py
Persistence + helpers for multiple named conversations.

Each conversation is a dict:
    {
        "id": {
            "name": str,
            "created": ISO-8601 timestamp,
            "updated": ISO-8601 timestamp,
            "messages": [ {"role": "user"|"assistant", "content": str, "sources": [...]?} ]
        }
    }

Conversations are persisted as JSON alongside the vectorstore data so histories
survive app restarts. This module is pure data — no Streamlit dependency — so it
mirrors chat.py's plain-module style and can be unit-tested in isolation.
"""

import json
import os
import uuid
from datetime import datetime
from typing import Dict, Any

DATA_DIR = os.path.join("data")
CONVERSATIONS_FILE = os.path.join(DATA_DIR, "chat_history.json")


def load_conversations() -> Dict[str, Any]:
    """Load persisted conversations. Returns {} if the file is missing/corrupt."""
    if not os.path.exists(CONVERSATIONS_FILE):
        return {}
    try:
        with open(CONVERSATIONS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        # Don't crash the app on a corrupt history file; drop to a starting point.
        return {}


def save_conversations(convos: Dict[str, Any]) -> None:
    """Persist all conversations to disk atomically-ish."""
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(CONVERSATIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(convos, f, ensure_ascii=False, indent=2)


def new_conversation_id() -> str:
    """Return a short unique id for a new conversation."""
    return uuid.uuid4().hex[:12]


def _now() -> str:
    return datetime.now().isoformat()


def touch(convo: Dict[str, Any]) -> None:
    """Bump the 'updated' timestamp (most-recent-first ordering)."""
    convo["updated"] = _now()


def create_conversation(convos: Dict[str, Any], name: str = "") -> Dict[str, Any]:
    """Create a conversation in-place and return it."""
    cid = new_conversation_id()
    ts = _now()
    convo = {
        "name": name or next_name(convos),
        "created": ts,
        "updated": ts,
        "messages": [],
    }
    convos[cid] = convo
    return convo


def next_name(convos: Dict[str, Any]) -> str:
    """Pick "New chat", "New chat 2", ... avoiding collisions with existing names."""
    taken = {c.get("name", "") for c in convos.values()}
    if "New conversation" not in taken:
        return "New conversation"
    n = 2
    while f"New conversation {n}" in taken:
        n += 1
    return f"New conversation {n}"


def ordered_ids(convos: Dict[str, Any]) -> list:
    """Return conversation ids most-recent-first by `updated`."""
    return sorted(convos, key=lambda c: convos[c].get("updated", ""), reverse=True)