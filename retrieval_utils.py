"""Defensive helpers for retrieval query inputs."""


def normalize_hyde_query(expanded_query: object, original_query: str) -> str:
    """Use the original query when HyDE returns None, a non-string, or blank text."""
    if not isinstance(expanded_query, str) or not expanded_query.strip():
        return original_query
    return expanded_query.strip()


def is_simple_query(question: str) -> bool:
    """Return True for short, single-intent questions that need no expansion."""
    if not isinstance(question, str):
        return False
    normalized = " ".join(question.lower().split())
    words = normalized.split()
    complex_markers = (
        " compare ", " comparison ", " difference ", " versus ", " vs ",
        " pros and cons ", " step by step ", " relationship ",
        " explain how ", " explain why ", " multiple ", " each of ",
        " across ", " both ", " and ", " or ",
    )
    padded = f" {normalized} "
    return len(words) <= 16 and len(question) <= 120 and not any(
        marker in padded for marker in complex_markers
    )
