from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - only used when tqdm is absent.
    def tqdm(iterable, **_: object):
        return iterable

from .data import save_json, save_jsonl, save_triples
from .embeddings import TripleEmbedder
from .llm_backends import LLMBackend
from .metrics import binary_metrics
from .noise import inject_entity_replacement_noise
from .retrieval import ContextIndex, filter_paper_candidates
from .types import DetectionRecord, RefinementRecord, Triple


@dataclass(frozen=True)
class RunConfig:
    noise: float
    seed: int
    use_context: bool = True
    context_k: int = 1
    keep_original_on_refine_failure: bool = False


def run_pipeline(
    clean_triples: list[Triple],
    *,
    backend: LLMBackend,
    embedder: TripleEmbedder,
    config: RunConfig,
    output_dir: str | Path,
    realize: Callable[[Triple], str] | None = None,
    run_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the paper's LLM_sim detection and refinement pipeline."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labeled = inject_entity_replacement_noise(
        clean_triples,
        noise_ratio=config.noise,
        seed=config.seed,
    )
    noisy_triples = [item.triple for item in labeled]
    noisy_index = ContextIndex(noisy_triples, embedder, realize=realize)
    clean_set = {triple.as_tuple() for triple in clean_triples}

    save_triples(output_dir / "clean_kg.tsv", clean_triples)
    save_triples(output_dir / "noisy_kg.tsv", noisy_triples)
    save_jsonl(output_dir / "noise_labels.jsonl", labeled)

    labels_correct: list[bool] = []
    predictions_correct: list[bool] = []
    renewed_triples: list[Triple] = []
    detections: list[DetectionRecord] = []
    refinements: list[RefinementRecord] = []

    iterator = tqdm(enumerate(labeled), total=len(labeled), desc="LLM_sim", unit="triple")
    for index, item in iterator:
        context_hit = None
        if config.use_context:
            hits = noisy_index.search(
                item.triple,
                top_k=config.context_k,
                exclude_indices={index},
            )
            context_hit = hits[0] if hits else None
        additional_context = context_hit.triple if context_hit else None

        detection = backend.detect(item.triple, additional_context)
        gold_is_correct = not item.is_noisy
        predicted_is_correct = detection.is_correct

        labels_correct.append(gold_is_correct)
        predictions_correct.append(predicted_is_correct)
        detections.append(
            DetectionRecord(
                index=index,
                input_triple=item.triple,
                clean_source=item.clean_source,
                gold_is_correct=gold_is_correct,
                predicted_is_correct=predicted_is_correct,
                final_answer=detection.final_answer,
                raw_text=detection.raw_text,
                additional_context=additional_context,
                additional_context_score=context_hit.score if context_hit else None,
                prompt=detection.prompt,
            )
        )

        if predicted_is_correct:
            renewed_triples.append(item.triple)
            continue

        candidate_output = backend.generate_candidates(item.triple)
        constrained = filter_paper_candidates(item.triple, candidate_output.candidates)
        selected_hit = noisy_index.select_refinement(item.triple, constrained)
        selected = selected_hit.triple if selected_hit else None

        if selected is not None:
            renewed_triples.append(selected)
        elif config.keep_original_on_refine_failure:
            renewed_triples.append(item.triple)

        refinements.append(
            RefinementRecord(
                index=index,
                input_triple=item.triple,
                clean_source=item.clean_source,
                raw_candidates=candidate_output.candidates,
                constrained_candidates=constrained,
                selected=selected,
                selected_similarity_score=selected_hit.score if selected_hit else None,
                selected_in_clean_kg=bool(selected and selected.as_tuple() in clean_set),
                raw_text=candidate_output.raw_text,
                prompt=candidate_output.prompt,
            )
        )

    metrics = binary_metrics(labels_correct, predictions_correct)
    selected_refinements = [record for record in refinements if record.selected is not None]
    refinement_correctness = (
        sum(record.selected_in_clean_kg for record in selected_refinements) / len(selected_refinements)
        if selected_refinements
        else 0.0
    )

    result: dict[str, Any] = {
        "paper": "Dong et al. 2025, Refining Noisy Knowledge Graph with Large Language Models",
        "config": {
            "noise": config.noise,
            "seed": config.seed,
            "use_context": config.use_context,
            "context_k": config.context_k,
            "keep_original_on_refine_failure": config.keep_original_on_refine_failure,
            "embedding_model": embedder.model_name,
            "embedding_fallback_used": embedder.using_fallback,
        },
        "metadata": run_metadata or {},
        "counts": {
            "clean_triples": len(clean_triples),
            "noisy_triples": len(noisy_triples),
            "true_noisy": sum(item.is_noisy for item in labeled),
            "predicted_noisy": sum(not prediction for prediction in predictions_correct),
            "renewed_triples": len(renewed_triples),
            "refinement_attempts": len(refinements),
            "successful_refinements": len(selected_refinements),
        },
        "detection": metrics,
        "refinement": {
            "correctness_against_clean_kg": refinement_correctness,
            "note": "The paper also reports manual human evaluation over 100 refined triples.",
        },
    }

    save_triples(output_dir / "renewed_kg.tsv", renewed_triples)
    save_jsonl(output_dir / "detection_records.jsonl", detections)
    save_jsonl(output_dir / "refinement_records.jsonl", refinements)
    save_json(output_dir / "metrics.json", result)
    return result


def run_pipeline_from_noisy(
    clean_triples: list[Triple],
    noisy_triples: list[Triple],
    *,
    backend: LLMBackend,
    embedder: TripleEmbedder,
    config: RunConfig,
    output_dir: str | Path,
    realize: Callable[[Triple], str] | None = None,
    run_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run LLM_sim when noisy triples have already been created externally.

    This is useful for DBP15K/DBP100K workflows where a separate script creates
    noisy_triples_1 and noisy_triples_2. The method still uses only the noisy KG
    for context retrieval/refinement; clean triples are kept for labels and
    correctness evaluation.
    """
    if len(clean_triples) != len(noisy_triples):
        raise ValueError(
            "clean_triples and noisy_triples must have the same length because "
            "labels are inferred row-by-row."
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    noisy_index = ContextIndex(noisy_triples, embedder, realize=realize)
    clean_set = {triple.as_tuple() for triple in clean_triples}

    save_triples(output_dir / "clean_kg.tsv", clean_triples)
    save_triples(output_dir / "noisy_kg.tsv", noisy_triples)

    labels_correct: list[bool] = []
    predictions_correct: list[bool] = []
    renewed_triples: list[Triple] = []
    detections: list[DetectionRecord] = []
    refinements: list[RefinementRecord] = []
    inferred_labels = []

    iterator = tqdm(
        enumerate(zip(clean_triples, noisy_triples)),
        total=len(noisy_triples),
        desc="LLM_sim",
        unit="triple",
    )
    for index, (clean_source, noisy_triple) in iterator:
        is_noisy = clean_source != noisy_triple
        inferred_labels.append(
            {
                "index": index,
                "triple": noisy_triple,
                "is_noisy": is_noisy,
                "clean_source": clean_source,
                "corruption_side": infer_corruption_side(clean_source, noisy_triple),
            }
        )

        context_hit = None
        if config.use_context:
            hits = noisy_index.search(
                noisy_triple,
                top_k=config.context_k,
                exclude_indices={index},
            )
            context_hit = hits[0] if hits else None
        additional_context = context_hit.triple if context_hit else None

        detection = backend.detect(noisy_triple, additional_context)
        gold_is_correct = not is_noisy
        predicted_is_correct = detection.is_correct

        labels_correct.append(gold_is_correct)
        predictions_correct.append(predicted_is_correct)
        detections.append(
            DetectionRecord(
                index=index,
                input_triple=noisy_triple,
                clean_source=clean_source,
                gold_is_correct=gold_is_correct,
                predicted_is_correct=predicted_is_correct,
                final_answer=detection.final_answer,
                raw_text=detection.raw_text,
                additional_context=additional_context,
                additional_context_score=context_hit.score if context_hit else None,
                prompt=detection.prompt,
            )
        )

        if predicted_is_correct:
            renewed_triples.append(noisy_triple)
            continue

        candidate_output = backend.generate_candidates(noisy_triple)
        constrained = filter_paper_candidates(noisy_triple, candidate_output.candidates)
        selected_hit = noisy_index.select_refinement(noisy_triple, constrained)
        selected = selected_hit.triple if selected_hit else None

        if selected is not None:
            renewed_triples.append(selected)
        elif config.keep_original_on_refine_failure:
            renewed_triples.append(noisy_triple)

        refinements.append(
            RefinementRecord(
                index=index,
                input_triple=noisy_triple,
                clean_source=clean_source,
                raw_candidates=candidate_output.candidates,
                constrained_candidates=constrained,
                selected=selected,
                selected_similarity_score=selected_hit.score if selected_hit else None,
                selected_in_clean_kg=bool(selected and selected.as_tuple() in clean_set),
                raw_text=candidate_output.raw_text,
                prompt=candidate_output.prompt,
            )
        )

    metrics = binary_metrics(labels_correct, predictions_correct)
    selected_refinements = [record for record in refinements if record.selected is not None]
    refinement_correctness = (
        sum(record.selected_in_clean_kg for record in selected_refinements) / len(selected_refinements)
        if selected_refinements
        else 0.0
    )

    result: dict[str, Any] = {
        "paper": "Dong et al. 2025, Refining Noisy Knowledge Graph with Large Language Models",
        "config": {
            "noise": config.noise,
            "seed": config.seed,
            "use_context": config.use_context,
            "context_k": config.context_k,
            "keep_original_on_refine_failure": config.keep_original_on_refine_failure,
            "embedding_model": embedder.model_name,
            "embedding_fallback_used": embedder.using_fallback,
            "noise_source": "external_noisy_triples",
        },
        "metadata": run_metadata or {},
        "counts": {
            "clean_triples": len(clean_triples),
            "noisy_triples": len(noisy_triples),
            "true_noisy": sum(clean != noisy for clean, noisy in zip(clean_triples, noisy_triples)),
            "predicted_noisy": sum(not prediction for prediction in predictions_correct),
            "renewed_triples": len(renewed_triples),
            "refinement_attempts": len(refinements),
            "successful_refinements": len(selected_refinements),
        },
        "detection": metrics,
        "refinement": {
            "correctness_against_clean_kg": refinement_correctness,
            "note": "The paper also reports manual human evaluation over 100 refined triples.",
        },
    }

    save_jsonl(output_dir / "noise_labels.jsonl", inferred_labels)
    save_triples(output_dir / "renewed_kg.tsv", renewed_triples)
    save_jsonl(output_dir / "detection_records.jsonl", detections)
    save_jsonl(output_dir / "refinement_records.jsonl", refinements)
    save_json(output_dir / "metrics.json", result)
    return result


def infer_corruption_side(clean: Triple, noisy: Triple) -> str | None:
    if clean == noisy:
        return None
    changed_head = clean.head != noisy.head
    changed_tail = clean.tail != noisy.tail
    changed_relation = clean.relation != noisy.relation
    if changed_relation:
        return "relation_or_multiple"
    if changed_head and not changed_tail:
        return "head"
    if changed_tail and not changed_head:
        return "tail"
    return "multiple"
