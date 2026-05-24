from __future__ import annotations

import unittest

from llm_sim_replication.noise import inject_entity_replacement_noise
from llm_sim_replication.types import Triple


class NoiseInjectionTest(unittest.TestCase):
    def test_noise_replaces_one_entity_and_keeps_relation(self) -> None:
        triples = [
            Triple("a", "r", "b"),
            Triple("c", "r", "d"),
            Triple("e", "r", "f"),
            Triple("g", "r", "h"),
            Triple("i", "r", "j"),
        ]
        labeled = inject_entity_replacement_noise(triples, noise_ratio=0.4, seed=7)
        noisy = [item for item in labeled if item.is_noisy]

        self.assertEqual(len(noisy), 2)
        clean_set = {triple.as_tuple() for triple in triples}
        for item in noisy:
            self.assertEqual(item.triple.relation, item.clean_source.relation)
            self.assertNotIn(item.triple.as_tuple(), clean_set)
            changed_head = item.triple.head != item.clean_source.head
            changed_tail = item.triple.tail != item.clean_source.tail
            self.assertNotEqual(changed_head, changed_tail)


if __name__ == "__main__":
    unittest.main()
