from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from llm_sim_replication.data import load_triples
from llm_sim_replication.embeddings import TripleEmbedder
from llm_sim_replication.llm_backends import HeuristicOracleBackend
from llm_sim_replication.pipeline import RunConfig, run_pipeline


class PipelineTest(unittest.TestCase):
    def test_toy_pipeline_writes_expected_artifacts(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        triples = load_triples(repo_root / "data" / "toy_kg.tsv")
        embedder = TripleEmbedder(require_sentence_transformer=False)
        backend = HeuristicOracleBackend(triples)

        with tempfile.TemporaryDirectory() as tmp:
            result = run_pipeline(
                triples,
                backend=backend,
                embedder=embedder,
                config=RunConfig(noise=0.2, seed=13),
                output_dir=tmp,
            )
            out = Path(tmp)
            self.assertTrue((out / "clean_kg.tsv").exists())
            self.assertTrue((out / "noisy_kg.tsv").exists())
            self.assertTrue((out / "renewed_kg.tsv").exists())
            self.assertTrue((out / "detection_records.jsonl").exists())
            self.assertTrue((out / "refinement_records.jsonl").exists())
            self.assertTrue((out / "metrics.json").exists())
            self.assertEqual(result["counts"]["true_noisy"], 3)
            self.assertEqual(result["detection"]["f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
