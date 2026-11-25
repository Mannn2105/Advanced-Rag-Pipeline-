"""Offline tests for retrieval ranking and evidence provenance."""
import unittest
from types import SimpleNamespace

import pipeline


def hit(node_id, file_name, page_label, text, score=0.5):
    node = SimpleNamespace(
        node_id=node_id,
        metadata={"file_name": file_name, "page_label": page_label},
        text=text,
    )
    return SimpleNamespace(node=node, score=score)


class RetrievalProvenanceTests(unittest.TestCase):
    def test_fusion_deduplicates_nodes_and_combines_rankings(self):
        first = hit("guide-1", "PROJECT_GUIDE.md", "2", "FastAPI serves the app")
        second = hit("faq-1", "DEMO_FAQ.md", "1", "Qdrant stores vectors")
        fused = pipeline.rag_fusion([[first, second], [second, first]])

        self.assertEqual([item.node.node_id for item in fused], ["guide-1", "faq-1"])
        self.assertEqual(len({item.node.node_id for item in fused}), 2)

    def test_source_diversity_limits_repeated_chunks_from_one_file(self):
        hits = [
            hit("a1", "PROJECT_GUIDE.md", "1", "First passage"),
            hit("a2", "PROJECT_GUIDE.md", "2", "Second passage"),
            hit("b1", "DEMO_FAQ.md", "1", "Independent source"),
        ]

        selected = pipeline.enforce_source_diversity(hits, max_per_file=1)

        self.assertEqual(
            [(item.node.metadata["file_name"], item.node.node_id) for item in selected],
            [("PROJECT_GUIDE.md", "a1"), ("DEMO_FAQ.md", "b1")],
        )

    def test_context_and_source_records_preserve_filename_and_page(self):
        selected = [
            hit("guide-1", "PROJECT_GUIDE.md", "3", "Qdrant stores vector embeddings.", 0.91),
            hit("faq-1", "DEMO_FAQ.md", "2", "The app uses shared HTTP Basic login.", 0.82),
        ]

        context = pipeline.format_context(selected)
        sources = pipeline._source_records(selected)

        self.assertIn("[SOURCE: PROJECT_GUIDE.md | PAGE: 3]", context)
        self.assertIn("[SOURCE: DEMO_FAQ.md | PAGE: 2]", context)
        self.assertEqual(
            sources,
            [
                {"file": "PROJECT_GUIDE.md", "page": "3", "score": 0.91},
                {"file": "DEMO_FAQ.md", "page": "2", "score": 0.82},
            ],
        )


if __name__ == "__main__":
    unittest.main()
