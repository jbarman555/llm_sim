from __future__ import annotations

import random
from collections.abc import Sequence

from .types import LabeledTriple, Triple


def inject_entity_replacement_noise(
    triples: Sequence[Triple],
    noise_ratio: float,
    seed: int,
) -> list[LabeledTriple]:
    """Create the paper's artificial noisy KG.

    For selected triples, replace either the head or tail entity while keeping
    the relation unchanged:

    (h, r, t) -> (h', r, t) or (h, r, t')

    The paper says the replacement entity should not relate to the fixed entity
    under relation r. In code, the observable constraint is that the corrupted
    triple must not already exist in the original clean KG.
    """
    if not 0 <= noise_ratio <= 1:
        raise ValueError("noise_ratio must be between 0 and 1.")
    if not triples:
        raise ValueError("Cannot inject noise into an empty KG.")

    rng = random.Random(seed)
    entities = sorted({triple.head for triple in triples} | {triple.tail for triple in triples})
    clean_set = {triple.as_tuple() for triple in triples}
    noise_count = int(round(len(triples) * noise_ratio))
    noisy_indices = set(rng.sample(range(len(triples)), noise_count))

    labeled: list[LabeledTriple] = []
    for index, triple in enumerate(triples):
        if index not in noisy_indices:
            labeled.append(
                LabeledTriple(
                    triple=triple,
                    is_noisy=False,
                    clean_source=triple,
                    corruption_side=None,
                )
            )
            continue

        corrupt_head = rng.random() < 0.5
        side = "head" if corrupt_head else "tail"
        corrupted = _sample_corruption(triple, entities, clean_set, corrupt_head, rng)
        labeled.append(
            LabeledTriple(
                triple=corrupted,
                is_noisy=True,
                clean_source=triple,
                corruption_side=side,
            )
        )
    return labeled


def _sample_corruption(
    triple: Triple,
    entities: Sequence[str],
    clean_set: set[tuple[str, str, str]],
    corrupt_head: bool,
    rng: random.Random,
) -> Triple:
    for _ in range(10_000):
        entity = rng.choice(entities)
        candidate = (
            Triple(entity, triple.relation, triple.tail)
            if corrupt_head
            else Triple(triple.head, triple.relation, entity)
        )
        if candidate != triple and candidate.as_tuple() not in clean_set:
            return candidate
    raise RuntimeError(
        "Unable to sample a corrupted triple that is absent from the clean KG. "
        "Try a larger graph or a lower noise ratio."
    )
