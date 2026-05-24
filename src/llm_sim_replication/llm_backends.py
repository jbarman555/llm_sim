from __future__ import annotations

import re
from collections.abc import Sequence

from .prompts import detection_prompt, refinement_prompt
from .types import CandidateOutput, DetectionOutput, Triple


class LLMBackend:
    """Interface for LLM_sim detection and refinement calls."""

    def detect(self, target: Triple, additional_context: Triple | None) -> DetectionOutput:
        raise NotImplementedError

    def generate_candidates(self, target: Triple) -> CandidateOutput:
        raise NotImplementedError


class HeuristicOracleBackend(LLMBackend):
    """Offline toy backend.

    This backend uses the clean KG set, so it must not be used for paper scores.
    It exists only to verify the rest of the pipeline without downloading LLMs.
    """

    def __init__(self, clean_triples: Sequence[Triple], *, save_prompts: bool = False) -> None:
        self.clean_set = {triple.as_tuple() for triple in clean_triples}
        self.clean_by_relation = list(clean_triples)
        self.save_prompts = save_prompts

    def detect(self, target: Triple, additional_context: Triple | None) -> DetectionOutput:
        prompt = detection_prompt(target, additional_context)
        is_correct = target.as_tuple() in self.clean_set
        answer = "yes" if is_correct else "no"
        raw = f"Final Answer: {answer}"
        return DetectionOutput(
            is_correct=is_correct,
            final_answer=answer,
            raw_text=raw,
            prompt=prompt if self.save_prompts else None,
        )

    def generate_candidates(self, target: Triple) -> CandidateOutput:
        prompt = refinement_prompt(target)
        candidates: list[Triple] = []
        for triple in self.clean_by_relation:
            if triple.relation != target.relation:
                continue
            if triple.head == target.head or triple.tail == target.tail:
                candidates.append(triple)
            if len(candidates) == 5:
                break
        if len(candidates) < 5:
            for triple in self.clean_by_relation:
                if triple.relation == target.relation and triple not in candidates:
                    candidates.append(triple)
                if len(candidates) == 5:
                    break
        raw = "\n".join(
            f"{index}. ({candidate.head}, {candidate.relation}, {candidate.tail})"
            for index, candidate in enumerate(candidates, start=1)
        )
        return CandidateOutput(
            candidates=candidates,
            raw_text=raw,
            prompt=prompt if self.save_prompts else None,
        )


class TransformersBackend(LLMBackend):
    """Hugging Face Transformers backend for paper-style Llama runs."""

    def __init__(
        self,
        model_name: str,
        *,
        max_new_tokens_detection: int = 128,
        max_new_tokens_refinement: int = 256,
        device_map: str = "auto",
        save_prompts: bool = False,
    ) -> None:
        try:
            from transformers import pipeline  # type: ignore
        except Exception as exc:
            raise RuntimeError("Install requirements.txt to use --backend hf.") from exc

        self.generator = pipeline(
            "text-generation",
            model=model_name,
            tokenizer=model_name,
            device_map=device_map,
        )
        self.max_new_tokens_detection = max_new_tokens_detection
        self.max_new_tokens_refinement = max_new_tokens_refinement
        self.save_prompts = save_prompts

    def detect(self, target: Triple, additional_context: Triple | None) -> DetectionOutput:
        prompt = detection_prompt(target, additional_context)
        raw = self._generate(prompt, max_new_tokens=self.max_new_tokens_detection)
        answer = parse_final_answer(raw)
        return DetectionOutput(
            is_correct=answer == "yes",
            final_answer=answer,
            raw_text=raw,
            prompt=prompt if self.save_prompts else None,
        )

    def generate_candidates(self, target: Triple) -> CandidateOutput:
        prompt = refinement_prompt(target)
        raw = self._generate(prompt, max_new_tokens=self.max_new_tokens_refinement)
        return CandidateOutput(
            candidates=parse_candidate_triples(raw, forced_relation=target.relation),
            raw_text=raw,
            prompt=prompt if self.save_prompts else None,
        )

    def _generate(self, prompt: str, *, max_new_tokens: int) -> str:
        kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "return_full_text": False,
        }
        eos_token_id = getattr(self.generator.tokenizer, "eos_token_id", None)
        if eos_token_id is not None:
            kwargs["pad_token_id"] = eos_token_id
        output = self.generator(prompt, **kwargs)
        return str(output[0]["generated_text"])


def parse_final_answer(text: str) -> str:
    """Parse yes/no from the required final-answer format."""
    lowered = text.lower()
    match = re.search(r"final\s+answer\s*:\s*(yes|no)\b", lowered)
    if match:
        return match.group(1)
    yes_pos = lowered.rfind("yes")
    no_pos = lowered.rfind("no")
    if yes_pos == -1 and no_pos == -1:
        return "no"
    return "yes" if yes_pos > no_pos else "no"


def parse_candidate_triples(text: str, *, forced_relation: str) -> list[Triple]:
    """Parse numbered '(head, relation, tail)' candidates from LLM output."""
    triples: list[Triple] = []
    seen: set[tuple[str, str, str]] = set()
    for match in re.finditer(r"\(([^,\n]+),([^,\n]+),([^)]+)\)", text):
        head = _clean_field(match.group(1))
        relation = _clean_field(match.group(2))
        tail = _clean_field(match.group(3))
        if relation != forced_relation:
            relation = forced_relation
        triple = Triple(head, relation, tail)
        if triple.as_tuple() in seen:
            continue
        triples.append(triple)
        seen.add(triple.as_tuple())
        if len(triples) == 5:
            break
    return triples


def _clean_field(value: str) -> str:
    return value.strip().strip("\"'` ")
