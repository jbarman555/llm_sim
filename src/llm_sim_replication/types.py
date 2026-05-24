from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Triple:
    """A knowledge graph triple."""

    head: str
    relation: str
    tail: str

    def realize(self) -> str:
        """Paper realization(E1, R, E2): convert a triple into text."""
        return f"{self.head} {self.relation} {self.tail}"

    def as_tuple(self) -> tuple[str, str, str]:
        return (self.head, self.relation, self.tail)


@dataclass(frozen=True)
class LabeledTriple:
    """A triple after artificial noise injection."""

    triple: Triple
    is_noisy: bool
    clean_source: Triple
    corruption_side: str | None


@dataclass(frozen=True)
class SearchHit:
    triple: Triple
    score: float
    index: int


@dataclass(frozen=True)
class DetectionOutput:
    is_correct: bool
    final_answer: str
    raw_text: str
    prompt: str | None = None


@dataclass(frozen=True)
class CandidateOutput:
    candidates: list[Triple]
    raw_text: str
    prompt: str | None = None


@dataclass(frozen=True)
class DetectionRecord:
    index: int
    input_triple: Triple
    clean_source: Triple
    gold_is_correct: bool
    predicted_is_correct: bool
    final_answer: str
    raw_text: str
    additional_context: Triple | None
    additional_context_score: float | None
    prompt: str | None


@dataclass(frozen=True)
class RefinementRecord:
    index: int
    input_triple: Triple
    clean_source: Triple
    raw_candidates: list[Triple]
    constrained_candidates: list[Triple]
    selected: Triple | None
    selected_similarity_score: float | None
    selected_in_clean_kg: bool
    raw_text: str
    prompt: str | None


def to_jsonable(value: Any) -> Any:
    """Convert dataclasses containing triples into JSON-serializable objects."""
    if isinstance(value, Triple):
        return {
            "head": value.head,
            "relation": value.relation,
            "tail": value.tail,
            "text": value.realize(),
        }
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "__dataclass_fields__"):
        return {key: to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    return value
