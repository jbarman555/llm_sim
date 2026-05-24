from __future__ import annotations

import argparse
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from .data import load_triples
from .embeddings import TripleEmbedder
from .llm_backends import HeuristicOracleBackend, LLMBackend, TransformersBackend
from .pipeline import RunConfig, run_pipeline_from_noisy
from .types import CandidateOutput, DetectionOutput, Triple


ENTITY_FILE_CANDIDATES = {
    1: ("ent_1", "ent_ids_1", "Ent_1", "entities_1"),
    2: ("ent_2", "ent_ids_2", "Ent_2", "entities_2"),
}

RELATION_FILE_CANDIDATES = {
    1: ("rel_1", "rel_ids_1", "Rel_1", "relations_1"),
    2: ("rel_2", "rel_ids_2", "Rel_2", "relations_2"),
}


@dataclass(frozen=True)
class LabelMap:
    raw_to_label: dict[str, str]
    label_to_raw: dict[str, str]

    @classmethod
    def empty(cls) -> "LabelMap":
        return cls({}, {})

    def label(self, raw: str) -> str:
        label = self.raw_to_label.get(raw, raw)
        if label == raw:
            return raw
        return f"{clean_label(label)} [ID={raw}]"

    def raw(self, value: str) -> str | None:
        value = value.strip().strip("\"'` ")
        id_match = re.search(r"\bID\s*=\s*([^\]\s]+)", value)
        if id_match:
            return id_match.group(1)
        if value in self.raw_to_label:
            return value
        if value in self.label_to_raw:
            return self.label_to_raw[value]
        return self.label_to_raw.get(normalize_label(value))


class DataFolderDisplayBackend(LLMBackend):
    """Show raw DBP entity ids as labels in LLM prompts, then map back to ids."""

    def __init__(
        self,
        inner: LLMBackend,
        entity_map: LabelMap,
        relation_map: LabelMap,
    ) -> None:
        self.inner = inner
        self.entity_map = entity_map
        self.relation_map = relation_map

    def detect(self, target: Triple, additional_context: Triple | None) -> DetectionOutput:
        return self.inner.detect(
            self.display_triple(target),
            self.display_triple(additional_context) if additional_context else None,
        )

    def generate_candidates(self, target: Triple) -> CandidateOutput:
        display_target = self.display_triple(target)
        raw_output = self.inner.generate_candidates(display_target)
        mapped: list[Triple] = []
        seen: set[tuple[str, str, str]] = set()

        for candidate in raw_output.candidates:
            head = self.entity_map.raw(candidate.head)
            tail = self.entity_map.raw(candidate.tail)
            if head is None or tail is None:
                continue
            triple = Triple(head, target.relation, tail)
            if triple.as_tuple() in seen:
                continue
            mapped.append(triple)
            seen.add(triple.as_tuple())

        return CandidateOutput(
            candidates=mapped,
            raw_text=raw_output.raw_text,
            prompt=raw_output.prompt,
        )

    def display_triple(self, triple: Triple | None) -> Triple | None:
        if triple is None:
            return None
        return Triple(
            self.entity_map.label(triple.head),
            self.relation_map.label(triple.relation),
            self.entity_map.label(triple.tail),
        )

    def realize(self, triple: Triple) -> str:
        display = self.display_triple(triple)
        assert display is not None
        return display.realize()


def run_dbp_data_directory(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    embedder = TripleEmbedder(
        args.embedding_model,
        require_sentence_transformer=args.require_sentence_transformer,
    )

    for side in (1, 2):
        clean_path = data_dir / f"triples_{side}"
        noisy_path = data_dir / f"noisy_triples_{side}"
        if not clean_path.exists():
            raise FileNotFoundError(f"Expected clean triple file: {clean_path}")
        if not noisy_path.exists():
            raise FileNotFoundError(
                f"Expected noisy triple file: {noisy_path}. "
                "Run scripts/inject_kg_noise.py first, or use scripts/run_dbp_data.sh."
            )

        clean_triples = load_triples(clean_path)
        noisy_triples = load_triples(noisy_path)
        entity_map = load_optional_label_map(data_dir, ENTITY_FILE_CANDIDATES[side])
        relation_map = load_optional_label_map(data_dir, RELATION_FILE_CANDIDATES[side])

        if args.backend == "heuristic":
            backend: LLMBackend = HeuristicOracleBackend(
                clean_triples,
                save_prompts=args.save_prompts,
            )
            realize = make_realizer(entity_map, relation_map)
        else:
            inner_backend = TransformersBackend(
                args.llm_model,
                max_new_tokens_detection=args.max_new_tokens_detection,
                max_new_tokens_refinement=args.max_new_tokens_refinement,
                device_map=args.device_map,
                save_prompts=args.save_prompts,
            )
            display_backend = DataFolderDisplayBackend(inner_backend, entity_map, relation_map)
            backend = display_backend
            realize = display_backend.realize

        run_dir = data_dir / args.run_dir_name / f"kg{side}"
        result = run_pipeline_from_noisy(
            clean_triples,
            noisy_triples,
            backend=backend,
            embedder=embedder,
            config=RunConfig(
                noise=args.noise,
                seed=args.seed,
                use_context=args.use_context,
                context_k=1,
                keep_original_on_refine_failure=args.keep_original_on_refine_failure,
            ),
            output_dir=run_dir,
            realize=realize,
            run_metadata={
                "source": {
                    "type": "dbp_data_folder",
                    "data_dir": str(data_dir),
                    "side": side,
                    "clean_triples": f"triples_{side}",
                    "noisy_triples": f"noisy_triples_{side}",
                },
                "output_file": f"llm_sim_triples_{side}",
            },
        )

        output_path = data_dir / f"llm_sim_triples_{side}"
        shutil.copyfile(run_dir / "renewed_kg.tsv", output_path)
        print(f"KG{side}: wrote {output_path}")
        print(
            f"KG{side}: true_noisy={result['counts']['true_noisy']} "
            f"predicted_noisy={result['counts']['predicted_noisy']} "
            f"f1={result['detection']['f1']:.4f}"
        )


def load_optional_label_map(data_dir: Path, candidates: tuple[str, ...]) -> LabelMap:
    for name in candidates:
        path = data_dir / name
        if path.exists():
            return load_label_map(path)
    return LabelMap.empty()


def load_label_map(path: Path) -> LabelMap:
    raw_to_label: dict[str, str] = {}
    label_to_raw: dict[str, str] = {}

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = re.split(r"\t|\s+", line, maxsplit=1)
            if len(parts) == 1:
                raw = parts[0]
                label = parts[0]
            else:
                first, second = parts[0].strip(), parts[1].strip()
                if looks_like_id(first):
                    raw, label = first, second
                elif looks_like_id(second):
                    raw, label = second, first
                else:
                    raw, label = first, second
            raw_to_label[raw] = label
            label_to_raw[label] = raw
            label_to_raw[clean_label(label)] = raw
            label_to_raw[normalize_label(label)] = raw

    return LabelMap(raw_to_label, label_to_raw)


def make_realizer(entity_map: LabelMap, relation_map: LabelMap):
    def realize(triple: Triple) -> str:
        return Triple(
            entity_map.label(triple.head),
            relation_map.label(triple.relation),
            entity_map.label(triple.tail),
        ).realize()

    return realize


def looks_like_id(value: str) -> bool:
    return value.isdigit()


def clean_label(value: str) -> str:
    value = value.strip().strip("<>")
    if "/" in value:
        value = value.rsplit("/", 1)[-1]
    if "#" in value:
        value = value.rsplit("#", 1)[-1]
    return value.replace("_", " ")


def normalize_label(value: str) -> str:
    value = clean_label(value)
    value = re.sub(r"\s*\[ID=[^\]]+\]", "", value)
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run LLM_sim on a DBP15K/DBP100K-style data directory."
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Directory containing triples_1, triples_2, noisy_triples_1, noisy_triples_2.",
    )
    parser.add_argument("--noise", type=float, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backend", choices=["heuristic", "hf"], default="hf")
    parser.add_argument("--llm-model", default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-new-tokens-detection", type=int, default=128)
    parser.add_argument("--max-new-tokens-refinement", type=int, default=256)
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument("--require-sentence-transformer", action="store_true")
    parser.add_argument(
        "--use-context",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use top-1 ADDITIONAL_CONTEXT from the noisy KG.",
    )
    parser.add_argument(
        "--keep-original-on-refine-failure",
        action="store_true",
        help="Keep detected noisy triples if no valid refinement maps back to KG ids.",
    )
    parser.add_argument("--save-prompts", action="store_true")
    parser.add_argument(
        "--run-dir-name",
        default="llm_sim_runs",
        help="Subdirectory for detailed JSONL/metrics artifacts.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_dbp_data_directory(args)


if __name__ == "__main__":
    main()
