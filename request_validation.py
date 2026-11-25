"""Input validation for public-facing RAG query requests."""
import os
import uuid

MAX_QUESTION_CHARS = int(os.getenv("MAX_QUESTION_CHARS", "4000"))
MAX_CONVERSATION_ID_CHARS = 128


def validate_query_payload(payload: object) -> tuple[str, str]:
    """Validate the JSON shape and bound user-controlled query identifiers."""
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object")

    question = payload.get("question")
    if not isinstance(question, str):
        raise ValueError("Question must be a string")

    question = question.strip()
    if not question:
        raise ValueError("Question cannot be empty")
    if len(question) > MAX_QUESTION_CHARS:
        raise ValueError(
            f"Question is too long (maximum {MAX_QUESTION_CHARS} characters)"
        )

    conversation_id = payload.get("conversation_id")
    if conversation_id is None:
        conversation_id = str(uuid.uuid4())
    if not isinstance(conversation_id, str) or not conversation_id.strip():
        raise ValueError("Conversation ID must be a non-empty string")
    if len(conversation_id) > MAX_CONVERSATION_ID_CHARS:
        raise ValueError(
            f"Conversation ID cannot exceed {MAX_CONVERSATION_ID_CHARS} characters"
        )

    return question, conversation_id
