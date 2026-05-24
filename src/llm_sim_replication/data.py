from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from .types import Triple, to_jsonable


HF_DATASETS = {
    "wn18rr": "VLyb/WN18RR",
    "fb15k-237": "VLyb/FB15k-237",
}


def load_triples(path: str | Path) -> list[Triple]:
    """Load triples from TSV or CSV.

    The first three columns are interpreted as head, relation, tail. A header
    row beginning with head/h/entity1 is skipped.
    """
    path = Path(path)
    triples: list[Triple] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        delimiter = "\t" if "\t" in sample else ","
        reader = csv.reader(handle, delimiter=delimiter)
        for row in reader:
            if len(row) < 3:
                continue
            first = row[0].strip().lower()
            if first in {"head", "h", "entity1", "subject"}:
                continue
            triples.append(Triple(row[0].strip(), row[1].strip(), row[2].strip()))
    if not triples:
        raise ValueError(f"No triples found in {path}")
    return triples


def save_triples(path: str | Path, triples: Iterable[Triple]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        for triple in triples:
            writer.writerow(triple.as_tuple())


def save_json(path: str | Path, obj: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(obj), indent=2), encoding="utf-8")


def save_jsonl(path: str | Path, rows: Iterable[object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(to_jsonable(row), ensure_ascii=False) + "\n")


def load_hf_triples(name: str, split: str) -> list[Triple]:
    """Load WN18RR or FB15k-237 from Hugging Face.

    The paper links to VLyb/WN18RR and VLyb/FB15k-237. This loader accepts
    common column layouts and falls back to the first three fields when needed.
    """
    if name not in HF_DATASETS:
        valid = ", ".join(sorted(HF_DATASETS))
        raise ValueError(f"Unknown dataset {name!r}. Valid options: {valid}")

    try:
        from datasets import load_dataset  # type: ignore
    except Exception as exc:
        raise RuntimeError("Install requirements.txt to use Hugging Face datasets.") from exc

    dataset = load_dataset(HF_DATASETS[name], split=split)
    triples: list[Triple] = []
    for row in dataset:
        head, relation, tail = _extract_triple_fields(row)
        triples.append(Triple(str(head), str(relation), str(tail)))
    if not triples:
        raise ValueError(f"No triples found in Hugging Face dataset {HF_DATASETS[name]}:{split}")
    return triples


def _extract_triple_fields(row: dict) -> tuple[object, object, object]:
    layouts = [
        ("head", "relation", "tail"),
        ("h", "r", "t"),
        ("head_id", "relation_id", "tail_id"),
        ("entity1", "relationship", "entity2"),
        ("entity1", "relation", "entity2"),
        ("subject", "predicate", "object"),
    ]
    for h_key, r_key, t_key in layouts:
        if h_key in row and r_key in row and t_key in row:
            return row[h_key], row[r_key], row[t_key]
    values = list(row.values())
    if len(values) < 3:
        raise ValueError(f"Cannot infer triple fields from row: {row}")
    return values[0], values[1], values[2]
