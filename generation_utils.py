"""Helpers for validating generated answer text."""


def normalize_generated_answer(answer: object) -> str:
    """Return trimmed answer text, or an empty string for unusable model output."""
    if not isinstance(answer, str):
        return ""
    return answer.strip()
