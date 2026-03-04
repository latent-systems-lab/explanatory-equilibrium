# Quick Start Guide

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[openai,dev]
cp .env.example .env
# fill keys in .env
set -a; source .env; set +a
```

Set API key (canonical default provider is OpenAI) by editing `.env` and then loading it:

```bash
set -a; source .env; set +a
```

## Sanity Checks

```bash
# Mock-only quick sanity (no API calls)
./scripts/sanity_check.sh

# OpenAI tiny grid sanity (matches canonical provider)
./scripts/sanity_all_openai.sh

# Optional Gemini parity sanity
set -a; source .env; set +a
./scripts/sanity_all_gemini.sh
```

## Full Repro Sweep (OpenAI)

```bash
./scripts/openai_full.sh
```

Canonical cache path used by scripts/config: `outputs/llm_cache_openai.jsonl`.
`outputs/` is intentionally not committed; regenerate results locally.

This generates a timestamped `outputs/final_paper_run_openai_*` folder containing:

- `logs.csv`
- `summary.csv`
- `paper_snippet.md`
- `plots/*.png` and `plots/*.pdf`
- `command.txt`

## Reproducibility

Use `scripts/openai_full.sh` for the main paper/workshop sweep and `scripts/sanity_all_openai.sh` for a cheaper verification pass.
All generated logs, summaries, and plots are written under `outputs/` and should be treated as local artifacts.

## Dev-only Utilities (non-essential)

These are not part of the core submission pipeline:

- `scripts/dev/gemini_api_debug.py`
- `scripts/dev/smoke_fixes.py`

Run them manually if needed:

```bash
python scripts/dev/gemini_api_debug.py
python scripts/dev/smoke_fixes.py
```
