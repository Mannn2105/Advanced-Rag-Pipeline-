"""Input validation for public-facing RAG query requests."""
import os
import uuid


def _positive_int(raw_value: str | None, name: str, default: int) -> int:
    """Parse a positive integer environment setting with a clear error."""
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


MAX_QUESTION_CHARS = _positive_int(os.getenv("MAX_QUESTION_CHARS"), "MAX_QUESTION_CHARS", 4000)


def validate_conversation_id(value: object) -> str:
    """Require a canonical UUID so IDs are safe in routes and UI attributes."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Conversation ID must be a valid UUID")
    try:
        parsed = uuid.UUID(value.strip())
    except (ValueError, AttributeError, TypeError) as exc:
        raise ValueError("Conversation ID must be a valid UUID") from exc
    return str(parsed)


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
    conversation_id = validate_conversation_id(conversation_id)

    return question, conversation_id
