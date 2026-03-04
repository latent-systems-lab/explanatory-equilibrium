#!/usr/bin/env bash
set -euo pipefail

# Canonical provider for artifact runs: OpenAI (matches paper results).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "Missing OPENAI_API_KEY." >&2
  exit 1
fi

# Quick dependency check (keeps failure readable)
python - <<'PY'
try:
    import openai  # noqa: F401
except Exception as e:
    raise SystemExit(
    "OpenAI SDK not importable. Install with: python -m pip install -e '.[openai]'\n" + str(e)
    )
print("OpenAI SDK import OK")  
PY

RUN_TS="$(date +%Y%m%d_%H%M%S)"

AMBIG_RATES=(0 0.5)

BASE_OUTDIR="outputs/sanity_all_openai_${RUN_TS}"

for ar in "${AMBIG_RATES[@]}"; do
  OUTDIR="${BASE_OUTDIR}/ambiguous_rate_${ar}"
  python -m ee_fin.cli run \
    --outdir "$OUTDIR" \
    --episodes 1 \
    --seeds 0 \
    --qs 0,0.5,1.0 \
    --max_words 30,60 \
    --ambiguous_rate "$ar" \
    --senders llm,llm_no_expl \
    --llm_provider openai \
    --model gpt-4.1 \
    --workers 1 \
    --debug
  echo "Wrote: $OUTDIR"
done

echo "Done. Base outdir: $BASE_OUTDIR"
