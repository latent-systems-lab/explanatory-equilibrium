#!/usr/bin/env bash
set -euo pipefail

# Alternate provider sanity check (not paper-default).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -z "${GEMINI_API_KEY:-}" && -z "${GOOGLE_API_KEY:-}" ]]; then
  echo "Missing GEMINI_API_KEY (or GOOGLE_API_KEY)." >&2
  exit 1
fi

OUTDIR="outputs/sanity_all_gemini_$(date +%Y%m%d_%H%M%S)"

python -m ee_fin.cli run \
  --outdir "$OUTDIR" \
  --episodes 2 \
  --seeds 0 \
  --qs 0,0.5,1.0 \
  --max_words 30,60 \
  --ambiguous_rate 0.5 \
  --senders llm,llm_no_expl \
  --llm_provider gemini \
  --workers 1 \
  --debug

echo "Wrote: $OUTDIR"
