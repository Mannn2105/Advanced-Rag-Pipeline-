"""Helpers for validating generated answer text."""


def normalize_generated_answer(answer: object) -> str:
    """Return trimmed answer text, or an empty string for unusable model output."""
    if not isinstance(answer, str):
        return ""
    return answer.strip()


def generate_with_fallback(primary_generate, fallback_generate, primary_error=None):
    """Try the primary provider, then a configured fallback provider."""
    if primary_error is None:
        try:
            answer = primary_generate()
            if isinstance(answer, str) and answer.strip():
                return answer
            primary_error = RuntimeError("Primary provider returned an empty answer.")
        except Exception as exc:
            primary_error = exc

    try:
        answer = fallback_generate()
        if isinstance(answer, str) and answer.strip():
            return answer
        raise RuntimeError("Fallback provider returned an empty answer.")
    except Exception as fallback_error:
        raise RuntimeError(
            "Primary generation provider failed and Groq fallback also failed: "
            f"{type(primary_error).__name__}: {primary_error}; "
            f"fallback {type(fallback_error).__name__}: {fallback_error}"
        ) from fallback_error
