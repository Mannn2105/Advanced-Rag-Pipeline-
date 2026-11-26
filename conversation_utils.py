"""Small, dependency-free helpers for conversation handling."""

import uuid


def resolve_conversation_id(body: dict) -> str:
    """Return a usable conversation ID, generating one for missing/null values."""
    return body.get("conversation_id") or str(uuid.uuid4())
