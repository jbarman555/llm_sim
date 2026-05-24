# LLM_sim Paper Replication

This repository implements the core method from:

> Dong Na, Natthawut Kertkeidkachorn, Xin Liu, and Kiyoaki Shirai. 2025.
> **Refining Noisy Knowledge Graph with Large Language Models.**
> GenAIK 2025. <https://aclanthology.org/2025.genaik-1.9/>

The implementation focuses on the paper's proposed **LLM_sim** method:

1. inject artificial entity-replacement noise into a clean KG;
2. retrieve one semantically similar triple as `ADDITIONAL_CONTEXT`;
3. detect noisy triples with Llama-style prompting;
4. generate five candidate refined triples for detected noise;
5. select the best refinement using grouped cosine similarity;
6. output the renewed KG.

The repository is designed so you can run a toy example immediately, then run
WN18RR/FB15k-237 after installing dependencies and configuring model access.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

For Llama 3, log in to Hugging Face and make sure your account has access to
the model:

```bash
huggingface-cli login
```

## Quick Offline Test

This uses the toy KG and an oracle backend. The oracle backend is only for
testing that the pipeline works; it must not be used for paper scores.

```bash
./scripts/run_toy.sh
```

Artifacts are written to:

```text
runs/toy-20/
```

## Paper-Style Run

Run WN18RR with 10% noise and LLM_sim context:

```bash
python3 replicate_llm_sim.py \
  --dataset wn18rr \
  --split train \
  --noise 0.1 \
  --seed 13 \
  --backend hf \
  --llm-model meta-llama/Meta-Llama-3-8B \
  --require-sentence-transformer \
  --use-context \
  --context-k 1 \
  --output runs/wn18rr-noise-0.1-llm-sim
```

Run the six dataset/noise combinations:

```bash
./scripts/run_paper_matrix.sh
```

You can override the model:

```bash
MODEL=meta-llama/Meta-Llama-3-8B ./scripts/run_paper_matrix.sh
```

## DBP15K / DBP100K Data-Folder Run

For your DBP15K or DBP100K data directory containing:

```text
ent_1        optional entity map for KG1
ent_2        optional entity map for KG2
triples_1    KG1 triples
triples_2    KG2 triples
```

run:

```bash
./scripts/run_dbp_data.sh /path/to/DBP15K/zh_en 0.1
```

This script first runs:

```text
scripts/inject_kg_noise.py
```

to create tail-corrupted noisy triples. Then it runs LLM_sim on those noisy
files. This matches the workflow:

```text
triples_1 -> noisy_triples_1 -> llm_sim_triples_1
triples_2 -> noisy_triples_2 -> llm_sim_triples_2
```

or directly:

```bash
python3 scripts/inject_kg_noise.py \
  --data-dir /path/to/DBP15K/zh_en \
  --noise 0.1 \
  --mode tail \
  --avoid-existing \
  --save-maps

PYTHONPATH=src python3 -m llm_sim_replication.dbp_data \
  --data-dir /path/to/DBP15K/zh_en \
  --noise 0.1 \
  --backend hf \
  --llm-model meta-llama/Meta-Llama-3-8B \
  --require-sentence-transformer \
  --use-context
```

The runner processes `triples_1` and `triples_2` separately. It writes these
files back into the same dataset directory:

```text
noisy_triples_1
noisy_triples_2
llm_sim_triples_1
llm_sim_triples_2
```

The noise script also writes optional audit files when `--save-maps` is used:

```text
noise_map_1.tsv
noise_map_2.tsv
noise_summary.json
```

Detailed records are saved under:

```text
llm_sim_runs/kg1/
llm_sim_runs/kg2/
```

If `ent_1` and `ent_2` exist, entity ids are displayed to the LLM as:

```text
Entity Label [ID=raw_id]
```

The final saved triples keep the original raw entity/relation IDs, so you can
feed `llm_sim_triples_1` and `llm_sim_triples_2` into your downstream code. If
your files are named `ent_ids_1` / `ent_ids_2`, the runner also recognizes
those names. The shell runner auto-detects `ent_ids_`, `ent_`, and `Ent_`; you
can override it with:

```bash
ENTITY_PREFIX=ent_ ./scripts/run_dbp_data.sh /path/to/dataset 0.1
```

## Exact Pipeline Implemented

### 1. Start From A Clean KG

The clean KG is a set of triples:

```text
G = {(h, r, t)}
```

For experiments, the paper uses:

- WN18RR
- FB15k-237

The clean KG is saved for labels and evaluation. It is not used as context
during the method.

### 2. Inject Noise

The paper injects 10%, 20%, and 30% noise. For a selected clean triple:

```text
(h, r, t)
```

replace either the head or the tail:

```text
(h', r, t)
(h, r, t')
```

The relation is kept unchanged. The replacement entity is sampled from the KG
entity set, and the resulting corrupted triple is rejected if it already exists
in the clean KG.

Code:

```text
src/llm_sim_replication/noise.py
```

### 3. Build ADDITIONAL_CONTEXT

Context is retrieved after noise injection from the current noisy KG. The paper
does not use the hidden clean KG for this step.

For each target triple:

```text
(E1, R, E2)
```

realize it as text:

```text
E1 R E2
```

Encode the target and every noisy KG triple with:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Compute cosine similarity and select the single most similar triple as:

```text
ADDITIONAL_CONTEXT
```

The current target row is excluded from its own search result. This avoids
trivially returning the query itself. The paper says it selects one similar
triple, so `--context-k 1` is the replication setting.

Code:

```text
src/llm_sim_replication/retrieval.py
```

### 4. Detect Noise

The LLM receives the target triple and the one retrieved context triple. It
must produce a final yes/no answer.

Interpretation:

```text
yes -> correct triple, keep it
no  -> noisy triple, send it to refinement
```

Detection metrics treat `correct/valid triple` as the positive class. This is
consistent with the paper's prompt and Table 2 pattern.

Code:

```text
src/llm_sim_replication/prompts.py
src/llm_sim_replication/llm_backends.py
```

### 5. Generate Refinement Candidates

For every detected noisy triple, the LLM generates five candidate triples.

The paper's constraints are enforced in code:

- keep the original relation;
- include either the original head or original tail;
- output up to five triples.

Example:

```text
Tokyo capital_of Paris
```

Valid candidates include:

```text
Tokyo capital_of Japan
France capital_of Paris
```

Invalid candidates include:

```text
Berlin capital_of Germany
Tokyo located_in Japan
```

Code:

```text
src/llm_sim_replication/llm_backends.py
src/llm_sim_replication/retrieval.py
```

### 6. Select The Best Refinement

Candidate triples are grouped against the current noisy KG using:

```text
(head, relation)
(relation, tail)
```

For each candidate, compute cosine similarity against grouped reference triples.
The selected refinement is:

```text
argmax candidate max reference cosine(candidate, reference)
```

The reference KG here is the noisy KG being cleaned. The clean KG is used only
for final correctness evaluation.

### 7. Build The Renewed KG

The final graph is:

```text
renewed KG = predicted-correct triples + selected refined triples
```

Artifacts:

```text
clean_kg.tsv
noisy_kg.tsv
noise_labels.jsonl
detection_records.jsonl
refinement_records.jsonl
renewed_kg.tsv
metrics.json
```

## Reproducing Paper Tables

### Table 2: Noise Detection

Run LLM_sim for:

```text
WN18RR:    10%, 20%, 30%
FB15k-237: 10%, 20%, 30%
```

Read:

```text
runs/*/metrics.json
```

Metrics included:

- accuracy
- precision
- recall
- F1

### Table 3: Noise Refinement

This repo computes:

```text
correctness_against_clean_kg
```

The paper also manually evaluates 100 random refined triples. That human
evaluation cannot be automated faithfully, so the JSONL output includes all
selected refinements for sampling.

### Table 4: KGC Task

This repository outputs the renewed KG needed for KGC:

```text
renewed_kg.tsv
```

The paper then trains/evaluates ExpressivE for KGC. The paper does not provide
the exact ExpressivE training configuration, random seeds, or official code, so
this repo intentionally does not fake those numbers. Use the generated KG files
as input to your chosen ExpressivE implementation and report the exact training
configuration you use.

## Tests

```bash
./scripts/run_tests.sh
```

## Important Reproducibility Notes

The paper does not specify every engineering detail. This repo makes the
following explicit choices:

- random seed is configurable with `--seed`;
- LLM decoding uses `do_sample=False`;
- target row is excluded from its own context search;
- generated candidates that violate the paper constraints are filtered out;
- failed refinements are dropped unless `--keep-original-on-refine-failure` is set;
- clean KG is never used for context retrieval or refinement selection.

These choices are conservative and auditable. Every intermediate record is
saved so you can inspect exactly what happened for each triple.
