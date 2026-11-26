"""Defensive helpers for retrieval query inputs."""


def normalize_hyde_query(expanded_query: object, original_query: str) -> str:
    """Use the original query when HyDE returns None, a non-string, or blank text."""
    if not isinstance(expanded_query, str) or not expanded_query.strip():
        return original_query
    return expanded_query.strip()
