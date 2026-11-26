"""Dependency-light helpers for local cross-encoder reranking."""


def local_cross_encoder_rerank(nodes, query, encoder, top_n):
    """Return the highest-scoring nodes using a supplied cross-encoder."""
    if not nodes or top_n <= 0:
        return []
    pairs = [(query, item.node.text) for item in nodes]
    scores = encoder.predict(pairs, show_progress_bar=False)
    if len(scores) != len(nodes):
        raise ValueError("Cross-encoder returned a score count that does not match the nodes.")
    ranked = sorted(
        zip(nodes, scores),
        key=lambda pair: float(pair[1]),
        reverse=True,
    )
    return [item for item, _ in ranked[:top_n]]
