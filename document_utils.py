"""Read-only catalog and diagnostics for text recovered from the persisted RAG node snapshot."""
import hashlib
import os


def _unwrap_node(item):
    return getattr(item, "node", item)


def _node_metadata(node):
    metadata = getattr(node, "metadata", None)
    return metadata if isinstance(metadata, dict) else {}


def _document_key(metadata):
    value = (
        metadata.get("file_path")
        or metadata.get("file_name")
        or metadata.get("filename")
        or metadata.get("source")
        or "Recovered document"
    )
    return str(value)


def _document_name(key):
    # Normalize both POSIX and Windows separators; never use this as a filesystem path.
    return key.replace("\\", "/").rstrip("/").split("/")[-1] or "Recovered document"


def _document_id(key):
    return hashlib.sha256(key.encode("utf-8", errors="replace")).hexdigest()[:16]


def _collect_documents(nodes):
    grouped = {}
    for item in nodes or []:
        node = _unwrap_node(item)
        metadata = _node_metadata(node)
        key = _document_key(metadata)
        entry = grouped.setdefault(key, {
            "name": _document_name(key), "chunks": [], "pages": set(), "seen": set()
        })
        text = getattr(node, "text", "") or ""
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


def _sort_pages(pages):
    return sorted(pages, key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value))


def build_document_catalog(nodes):
    grouped = _collect_documents(nodes)
    catalog = []
    for key, entry in grouped.items():
        catalog.append({
            "id": _document_id(key),
            "name": entry["name"],
            "chunk_count": len(entry["chunks"]),
            "pages": _sort_pages(entry["pages"]),
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
                "pages": _sort_pages(entry["pages"]),
            }
    return None


def build_document_diagnostics(nodes, snapshot_path):
    """Return safe index-health facts without exposing paths or document contents."""
    snapshot_exists = os.path.isfile(snapshot_path)
    snapshot_size = os.path.getsize(snapshot_path) if snapshot_exists else 0
    node_list = list(nodes or [])
    grouped = _collect_documents(node_list)
    return {
        "snapshot_exists": snapshot_exists,
        "snapshot_size_bytes": snapshot_size,
        "loaded_chunks": len(node_list),
        "document_count": len(grouped),
        "chunks_with_text": sum(bool(str(getattr(_unwrap_node(item), "text", "") or "").strip()) for item in node_list),
        "documents_with_recoverable_text": sum(bool(entry["chunks"]) for entry in grouped.values()),
        "metadata_keys": sorted({
            key
            for item in node_list
            for key in _node_metadata(_unwrap_node(item)).keys()
        }),
    }
