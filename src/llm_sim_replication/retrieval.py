from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence

import numpy as np

from .embeddings import TripleEmbedder
from .types import SearchHit, Triple


class ContextIndex:
    """Cosine-search index over realized KG triples."""

    def __init__(
        self,
        triples: Sequence[Triple],
        embedder: TripleEmbedder,
        *,
        realize: Callable[[Triple], str] | None = None,
    ) -> None:
        if not triples:
            raise ValueError("ContextIndex requires at least one triple.")
        self.triples = list(triples)
        self.embedder = embedder
        self.realize = realize or (lambda triple: triple.realize())
        self.embeddings = embedder.encode([self.realize(triple) for triple in self.triples])

    def search(
        self,
        query: Triple,
        *,
        top_k: int = 1,
        exclude_indices: Iterable[int] = (),
    ) -> list[SearchHit]:
        """Return top-k cosine hits for Section 3.1 context retrieval."""
        if top_k <= 0:
            return []

        excluded = set(exclude_indices)
        query_vec = self.embedder.encode([self.realize(query)])[0]
        scores = self.embeddings @ query_vec
        ranked = np.argsort(-scores)

        hits: list[SearchHit] = []
        for raw_index in ranked:
            index = int(raw_index)
            if index in excluded:
                continue
            hits.append(SearchHit(self.triples[index], float(scores[index]), index))
            if len(hits) == top_k:
                break
        return hits

    def select_refinement(
        self,
        noisy_triple: Triple,
        candidates: Sequence[Triple],
    ) -> SearchHit | None:
        """Select the best refinement candidate using Section 3.2.3.

        For each candidate Gj, compute the maximum cosine similarity to grouped
        reference triples Fi from the current KG, then choose the candidate with
        the highest such score.
        """
        candidates = filter_paper_candidates(noisy_triple, candidates)
        if not candidates:
            return None

        candidate_vectors = self.embedder.encode([self.realize(candidate) for candidate in candidates])
        best_candidate: Triple | None = None
        best_score = -math.inf

        for candidate, vector in zip(candidates, candidate_vectors):
            references = self.group_references(noisy_triple, candidate)
            if not references:
                continue
            reference_vectors = self.embedder.encode([self.realize(reference) for reference in references])
            score = float(np.max(reference_vectors @ vector))
            if score > best_score:
                best_score = score
                best_candidate = candidate

        if best_candidate is None:
            return None
        return SearchHit(best_candidate, best_score, index=-1)

    def group_references(self, noisy_triple: Triple, candidate: Triple) -> list[Triple]:
        """Reference groups from Section 3.2.2.

        The generated candidate must keep either the original head or original
        tail. If it keeps the original head, compare with KG triples sharing
        (head, relation). If it keeps the original tail, compare with triples
        sharing (relation, tail).
        """
        grouped: list[Triple] = []
        seen: set[tuple[str, str, str]] = set()

        if candidate.head == noisy_triple.head:
            for triple in self.triples:
                if triple.head == candidate.head and triple.relation == candidate.relation:
                    _append_unique(grouped, seen, triple)

        if candidate.tail == noisy_triple.tail:
            for triple in self.triples:
                if triple.relation == candidate.relation and triple.tail == candidate.tail:
                    _append_unique(grouped, seen, triple)

        return grouped


def filter_paper_candidates(noisy_triple: Triple, candidates: Sequence[Triple]) -> list[Triple]:
    """Enforce the refinement prompt constraints.

    A candidate must preserve the original relation and include either the
    original head or original tail.
    """
    filtered: list[Triple] = []
    seen: set[tuple[str, str, str]] = set()
    for candidate in candidates:
        if candidate == noisy_triple:
            continue
        if candidate.relation != noisy_triple.relation:
            continue
        if candidate.head != noisy_triple.head and candidate.tail != noisy_triple.tail:
            continue
        _append_unique(filtered, seen, candidate)
    return filtered


def _append_unique(items: list[Triple], seen: set[tuple[str, str, str]], triple: Triple) -> None:
    key = triple.as_tuple()
    if key in seen:
        return
    items.append(triple)
    seen.add(key)
