from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

from llm_sim_replication.dbp_data import load_label_map, run_dbp_data_directory


class DBPDataRunnerTest(unittest.TestCase):
    def test_label_map_accepts_id_first_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ent_1"
            path.write_text("1\thttp://dbpedia.org/resource/Tokyo\n", encoding="utf-8")
            label_map = load_label_map(path)

            self.assertEqual(label_map.raw("1"), "1")
            self.assertEqual(label_map.raw("Tokyo"), "1")
            self.assertIn("Tokyo", label_map.label("1"))

    def test_dbp_runner_writes_llm_sim_files_from_existing_noisy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "triples_1").write_text(
                "a\tr\tb\nc\tr\td\ne\tr\tf\ng\tr\th\ni\tr\tj\n",
                encoding="utf-8",
            )
            (root / "triples_2").write_text(
                "k\tr\tl\nm\tr\tn\no\tr\tp\nq\tr\ts\nt\tr\tu\n",
                encoding="utf-8",
            )
            (root / "noisy_triples_1").write_text(
                "a\tr\tc\nc\tr\td\ne\tr\tf\ng\tr\th\ni\tr\tj\n",
                encoding="utf-8",
            )
            (root / "noisy_triples_2").write_text(
                "k\tr\tm\nm\tr\tn\no\tr\tp\nq\tr\ts\nt\tr\tu\n",
                encoding="utf-8",
            )
            args = argparse.Namespace(
                data_dir=str(root),
                noise=0.2,
                seed=42,
                backend="heuristic",
                llm_model="unused",
                device_map="auto",
                max_new_tokens_detection=128,
                max_new_tokens_refinement=256,
                embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                require_sentence_transformer=False,
                use_context=True,
                keep_original_on_refine_failure=False,
                save_prompts=False,
                run_dir_name="llm_sim_runs",
            )

            run_dbp_data_directory(args)

            self.assertTrue((root / "llm_sim_triples_1").exists())
            self.assertTrue((root / "llm_sim_triples_2").exists())


if __name__ == "__main__":
    unittest.main()
