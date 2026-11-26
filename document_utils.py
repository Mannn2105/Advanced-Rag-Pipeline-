"""Utilities for cataloguing and reconstructing text from persisted RAG nodes.

These helpers deliberately operate only on the already-loaded nodes.pkl data. They
never read arbitrary filesystem paths and never modify the Qdrant collection.
"""
import hashlib
from pathlib import PurePath


def _node_metadata(node):
    metadata = getattr(node, "metadata", None)
    return metadata if isinstance(metadata, dict) else {}


def _document_key(metadata):
    return str(
        metadata.get("file_path")
        or metadata.get("file_name")
        or metadata.get("filename")
        or "Recovered document"
    )


def _document_id(key):
    return hashlib.sha256(key.encode("utf-8", errors="replace")).hexdigest()[:16]


def _collect_documents(nodes):
    grouped = {}
    for node in nodes or []:
        metadata = _node_metadata(getattr(node, "node", node))
        key = _document_key(metadata)
        entry = grouped.setdefault(key, {"name": PurePath(key).name, "chunks": [], "pages": set(), "seen": set()})
        text = getattr(getattr(node, "node", node), "text", "") or ""
        text = str(text).strip()
        if text:
            fingerprint = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
            if fingerprint not in entry["seen"]:
                entry["seen"].add(fingerprint)
                entry["chunks"].append(text)
        page = metadata.get("page_label") or metadata.get("page_number") or metadata.get("page")
        if page not in (None, ""):
            entry["pages"].add(str(page))
    return grouped


def build_document_catalog(nodes):
    grouped = _collect_documents(nodes)
    catalog = []
    for key, entry in grouped.items():
        catalog.append({
            "id": _document_id(key),
            "name": entry["name"],
            "chunk_count": len(entry["chunks"]),
            "pages": sorted(entry["pages"], key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value)),
            "reconstructed": True,
        })
    return sorted(catalog, key=lambda item: item["name"].lower())


def get_reconstructed_document(nodes, document_id):
    for key, entry in _collect_documents(nodes).items():
        if _document_id(key) == document_id:
            return {
                "id": document_id,
                "name": entry["name"],
                "text": "\n\n".join(entry["chunks"]),
                "chunk_count": len(entry["chunks"]),
                "pages": sorted(entry["pages"]),
            }
    return None
