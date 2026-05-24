from __future__ import annotations

import unittest

from llm_sim_replication.embeddings import TripleEmbedder
from llm_sim_replication.retrieval import ContextIndex, filter_paper_candidates
from llm_sim_replication.types import Triple


class RetrievalTest(unittest.TestCase):
    def test_additional_context_returns_one_non_self_hit(self) -> None:
        triples = [
            Triple("dog", "hypernym", "animal"),
            Triple("cat", "hypernym", "animal"),
            Triple("Paris", "capital_of", "France"),
        ]
        embedder = TripleEmbedder(require_sentence_transformer=False)
        index = ContextIndex(triples, embedder)

        hits = index.search(triples[0], top_k=1, exclude_indices={0})

        self.assertEqual(len(hits), 1)
        self.assertNotEqual(hits[0].index, 0)

    def test_candidate_constraints_match_paper_prompt(self) -> None:
        noisy = Triple("Tokyo", "capital_of", "Paris")
        candidates = [
            Triple("Tokyo", "capital_of", "Japan"),
            Triple("France", "capital_of", "Paris"),
            Triple("Berlin", "capital_of", "Germany"),
            Triple("Tokyo", "located_in", "Japan"),
        ]

        filtered = filter_paper_candidates(noisy, candidates)

        self.assertEqual(
            [candidate.as_tuple() for candidate in filtered],
            [
                ("Tokyo", "capital_of", "Japan"),
                ("France", "capital_of", "Paris"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
