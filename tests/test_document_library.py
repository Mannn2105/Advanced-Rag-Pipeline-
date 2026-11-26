import io
import os
import tempfile
import unittest
import zipfile
from types import SimpleNamespace

from document_utils import (
    build_document_catalog,
    build_document_diagnostics,
    get_reconstructed_document,
    recovered_download_details,
    build_recovered_documents_zip,
)


class DocumentLibraryTests(unittest.TestCase):
    def setUp(self):
        self.nodes = [
            SimpleNamespace(text="First chunk", metadata={"file_name": "notes.pdf", "page_label": "1"}),
            SimpleNamespace(text="Second chunk", metadata={"file_name": "notes.pdf", "page_label": "2"}),
            SimpleNamespace(text="First chunk", metadata={"file_name": "notes.pdf", "page_label": "1"}),
            SimpleNamespace(text="Other file text", metadata={"file_name": "guide.docx"}),
        ]

    def test_catalog_groups_files_and_deduplicates_chunks(self):
        catalog = build_document_catalog(self.nodes)
        self.assertEqual(len(catalog), 2)
        notes = next(item for item in catalog if item["name"] == "notes.pdf")
        self.assertEqual(notes["chunk_count"], 2)
        self.assertEqual(notes["pages"], ["1", "2"])

    def test_document_text_is_reconstructed(self):
        catalog = build_document_catalog(self.nodes)
        notes = next(item for item in catalog if item["name"] == "notes.pdf")
        recovered = get_reconstructed_document(self.nodes, notes["id"])
        self.assertIn("First chunk", recovered["text"])
        self.assertIn("Second chunk", recovered["text"])
        self.assertEqual(recovered["chunk_count"], 2)

    def test_windows_source_path_produces_safe_filename(self):
        nodes = [SimpleNamespace(text="Recovered", metadata={"file_path": r"C:\\private\\folder\\contract.pdf"})]
        self.assertEqual(build_document_catalog(nodes)[0]["name"], "contract.pdf")

    def test_node_with_score_style_wrapper_is_supported(self):
        wrapped = [SimpleNamespace(node=self.nodes[0])]
        catalog = build_document_catalog(wrapped)
        self.assertEqual(catalog[0]["name"], "notes.pdf")
        self.assertEqual(catalog[0]["chunk_count"], 1)

    def test_diagnostics_report_snapshot_and_metadata_without_content(self):
        with tempfile.NamedTemporaryFile() as snapshot:
            snapshot.write(b"snapshot")
            snapshot.flush()
            diagnostics = build_document_diagnostics(self.nodes, snapshot.name)
        self.assertTrue(diagnostics["snapshot_exists"])
        self.assertEqual(diagnostics["snapshot_size_bytes"], 8)
        self.assertEqual(diagnostics["loaded_chunks"], 4)
        self.assertEqual(diagnostics["document_count"], 2)
        self.assertIn("file_name", diagnostics["metadata_keys"])
        self.assertNotIn("First chunk", str(diagnostics))

    def test_missing_snapshot_is_reported(self):
        diagnostics = build_document_diagnostics(self.nodes, "/not/a/real/nodes.pkl")
        self.assertFalse(diagnostics["snapshot_exists"])
        self.assertEqual(diagnostics["snapshot_size_bytes"], 0)

    def test_markdown_download_keeps_markdown_extension(self):
        filename, media_type = recovered_download_details("folder/guide.md")
        self.assertEqual(filename, "guide-recovered.md")
        self.assertEqual(media_type, "text/markdown; charset=utf-8")

    def test_non_markdown_download_uses_text_extension(self):
        filename, media_type = recovered_download_details(r"C:\\private\\report.pdf")
        self.assertEqual(filename, "report-recovered.txt")
        self.assertEqual(media_type, "text/plain; charset=utf-8")

    def test_bulk_archive_contains_recovered_documents_without_paths(self):
        archive_bytes = build_recovered_documents_zip(self.nodes)
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            self.assertEqual(set(archive.namelist()), {"notes-recovered.txt", "guide-recovered.txt"})
            self.assertIn("First chunk", archive.read("notes-recovered.txt").decode("utf-8"))
            self.assertNotIn("private", " ".join(archive.namelist()))

    def test_unknown_document_returns_none(self):
        self.assertIsNone(get_reconstructed_document(self.nodes, "missing-document"))


if __name__ == "__main__":
    unittest.main()
