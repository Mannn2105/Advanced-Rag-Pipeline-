"""Helpers for validating and normalizing generated answer text."""

import json
import re


def normalize_generated_answer(answer: object) -> str:
    """Return readable answer text, unwrapping common JSON answer envelopes."""
    if not isinstance(answer, str):
        return ""
    text = answer.strip()
    if not text:
        return ""

    # Some providers return a JSON object even when plain Markdown was requested.
    # Unwrap only a clear string-valued answer field; otherwise preserve the text.
    candidate = text
    if candidate.startswith("```") and candidate.endswith("```"):
        candidate = candidate.strip("`").strip()
        if candidate.lower().startswith("json"):
            candidate = candidate[4:].strip()
    if candidate.startswith("{") and candidate.endswith("}"):
        try:
            payload = json.loads(candidate)
            if isinstance(payload, dict) and isinstance(payload.get("answer"), str):
                text = payload["answer"].strip()
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    # Clean provider artifacts seen in generated Markdown: empty bullet lines
    # ending in a backslash, or a backslash on a line by itself.
    cleaned_lines = []
    for line in text.splitlines():
        if re.fullmatch(r"\\s*\\\\\\s*", line):
            continue
        if re.fullmatch(r"\\s*[-*+]\\s*\\\\\\s*", line):
            continue
        if re.match(r"^\\s*[-*+]\\s+.*\\\\\\s*$", line):
            line = re.sub(r"\\\\\\s*$", "", line)
        cleaned_lines.append(line)
    return "\\n".join(cleaned_lines).strip()


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
