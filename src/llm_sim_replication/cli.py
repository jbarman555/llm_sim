from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import load_hf_triples, load_triples
from .embeddings import TripleEmbedder
from .llm_backends import HeuristicOracleBackend, TransformersBackend
from .pipeline import RunConfig, run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run paper-faithful LLM_sim noisy KG detection/refinement."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="Local TSV/CSV with head, relation, tail columns.")
    source.add_argument("--dataset", choices=["wn18rr", "fb15k-237"], help="Hugging Face dataset.")

    parser.add_argument("--split", default="train", help="Dataset split for --dataset.")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit for quick debugging.")
    parser.add_argument("--noise", type=float, choices=[0.1, 0.2, 0.3], required=True)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output", required=True, help="Directory for all run artifacts.")

    parser.add_argument("--backend", choices=["heuristic", "hf"], default="hf")
    parser.add_argument("--llm-model", default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-new-tokens-detection", type=int, default=128)
    parser.add_argument("--max-new-tokens-refinement", type=int, default=256)

    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Paper uses sentence-transformers/all-MiniLM-L6-v2.",
    )
    parser.add_argument(
        "--require-sentence-transformer",
        action="store_true",
        help="Fail if the paper embedding model cannot be loaded.",
    )
    parser.add_argument(
        "--use-context",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Section 3.1 ADDITIONAL_CONTEXT. Use --no-use-context for ablation.",
    )
    parser.add_argument(
        "--context-k",
        type=int,
        default=1,
        help="Paper uses one most-similar context triple; leave this at 1 for replication.",
    )
    parser.add_argument(
        "--keep-original-on-refine-failure",
        action="store_true",
        help="By default failed refinements are dropped from renewed KG.",
    )
    parser.add_argument(
        "--save-prompts",
        action="store_true",
        help="Store full prompts in JSONL records for debugging/auditing.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.input:
        triples = load_triples(args.input)
        source = {"type": "local", "input": str(Path(args.input))}
    else:
        triples = load_hf_triples(args.dataset, args.split)
        source = {"type": "huggingface", "dataset": args.dataset, "split": args.split}

    if args.limit:
        triples = triples[: args.limit]

    embedder = TripleEmbedder(
        args.embedding_model,
        require_sentence_transformer=args.require_sentence_transformer,
    )

    if args.backend == "heuristic":
        backend = HeuristicOracleBackend(triples, save_prompts=args.save_prompts)
        backend_metadata = {
            "backend": "heuristic",
            "warning": "Oracle backend uses clean KG and is only for tests/toy runs.",
        }
    else:
        backend = TransformersBackend(
            args.llm_model,
            max_new_tokens_detection=args.max_new_tokens_detection,
            max_new_tokens_refinement=args.max_new_tokens_refinement,
            device_map=args.device_map,
            save_prompts=args.save_prompts,
        )
        backend_metadata = {
            "backend": "hf",
            "llm_model": args.llm_model,
            "decoding": "do_sample=False",
        }

    result = run_pipeline(
        triples,
        backend=backend,
        embedder=embedder,
        config=RunConfig(
            noise=args.noise,
            seed=args.seed,
            use_context=args.use_context,
            context_k=args.context_k,
            keep_original_on_refine_failure=args.keep_original_on_refine_failure,
        ),
        output_dir=args.output,
        run_metadata={
            "source": source,
            "backend": backend_metadata,
            "limit": args.limit or None,
        },
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
