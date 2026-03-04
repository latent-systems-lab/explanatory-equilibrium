# Explanatory Equilibrium (Finance) — Bounded Verification Pre-Experiment

This project simulates a toy, finance-themed coordination problem between a **Trader** (sender) and a **Risk Manager** (receiver/verifier).

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[openai,dev]
cp .env.example .env
# fill keys in .env (OPENAI_API_KEY and/or GEMINI_API_KEY)
set -a; source .env; set +a
```

## Running (Mock Sender)

```bash
python -m ee_fin.cli run \
  --outdir outputs/run_mock \
  --episodes 200 \
  --seed 123 \
  --qs 0,0.1,0.3,0.6,1.0 \
  --max_words 30,60,120 \
  --sender mock
```

## Running (LLM Sender)

Canonical default provider for this artifact is **OpenAI**.

```bash
set -a; source .env; set +a
python -m ee_fin.cli run \
  --outdir outputs/run_llm_openai \
  --episodes 50 \
  --seed 123 \
  --qs 0,0.3,1.0 \
  --max_words 60 \
  --sender llm \
  --llm_provider openai \
  --cache_path outputs/llm_cache_openai.jsonl \
  --model gpt-4.1
```

Gemini can still be used as an alternate provider:

```bash
set -a; source .env; set +a
python -m ee_fin.cli run \
  --outdir outputs/run_llm_gemini \
  --episodes 50 \
  --seed 123 \
  --qs 0,0.3,1.0 \
  --max_words 60 \
  --sender llm \
  --llm_provider gemini \
  --cache_path outputs/llm_cache_openai.jsonl \
  --model gemini-2.5-flash
```

## Reproducibility

**Paper-results note:** paper/workshop figures were generated with **OpenAI (`gpt-4.1`)** using `scripts/openai_full.sh`; the exact-results scripts are `scripts/openai_full.sh` (full run) and `scripts/sanity_all_openai.sh` (small verification sweep).

Run exact reproduction:

```bash
./scripts/openai_full.sh
```

Canonical cache path used by scripts/config: `outputs/llm_cache_openai.jsonl`.
The `outputs/` directory is intentionally not committed; regenerate results locally.

Expected output directory pattern:

- `outputs/final_paper_run_openai_<timestamp>/logs.csv`
- `outputs/final_paper_run_openai_<timestamp>/summary.csv`
- `outputs/final_paper_run_openai_<timestamp>/paper_snippet.md`
- `outputs/final_paper_run_openai_<timestamp>/plots/*.png`
- `outputs/final_paper_run_openai_<timestamp>/plots/*.pdf`
- `outputs/final_paper_run_openai_<timestamp>/command.txt`

Quick verification run:

```bash
./scripts/sanity_all_openai.sh
```

Expected sanity output pattern:

- `outputs/sanity_all_openai_<timestamp>/ambiguous_rate_0/logs.csv`
- `outputs/sanity_all_openai_<timestamp>/ambiguous_rate_0/summary.csv`
- `outputs/sanity_all_openai_<timestamp>/ambiguous_rate_0/paper_snippet.md`
- `outputs/sanity_all_openai_<timestamp>/ambiguous_rate_0/plots/*.png`
- `outputs/sanity_all_openai_<timestamp>/ambiguous_rate_0/plots/*.pdf`
- same files for `ambiguous_rate_0.5/`

## Non-essential developer scripts

To avoid test-discovery ambiguity, manual/debug scripts live under `scripts/dev/` and are not part of the core reproducibility pipeline:

- `scripts/dev/gemini_api_debug.py` (Gemini connectivity/JSON debug helper)
- `scripts/dev/smoke_fixes.py` (manual smoke script)
