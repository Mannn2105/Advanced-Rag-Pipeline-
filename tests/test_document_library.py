import unittest
from types import SimpleNamespace

from document_utils import build_document_catalog, get_reconstructed_document


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

    def test_document_text_is_reconstructed_and_downloadable(self):
        catalog = build_document_catalog(self.nodes)
        notes = next(item for item in catalog if item["name"] == "notes.pdf")
        recovered = get_reconstructed_document(self.nodes, notes["id"])
        self.assertIn("First chunk", recovered["text"])
        self.assertIn("Second chunk", recovered["text"])
        self.assertEqual(recovered["chunk_count"], 2)

    def test_unknown_document_returns_none(self):
        self.assertIsNone(get_reconstructed_document(self.nodes, "missing-document"))


if __name__ == "__main__":
    unittest.main()
