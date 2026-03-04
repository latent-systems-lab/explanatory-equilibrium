#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

###############################################################################
# FINAL PAPER RUN CONFIG (used for workshop/paper reproducibility)
###############################################################################

# Provider: openai or gemini
LLM_PROVIDER="openai"

# OpenAI model (used when LLM_PROVIDER=openai)
OPENAI_MODEL="gpt-4.1"

# Gemini model (used when LLM_PROVIDER=gemini)
GEMINI_MODEL="gemini-2.5-flash"

# Core experiment sweep
EPISODES=200
SEEDS="0,1,2,3,4"
QS="0,0.1,0.3,0.5,0.7,1.0"
MAX_WORDS="30,60,120"
SENDERS="llm,llm_no_expl"

# Probability an episode is in the ambiguous_hedge regime
AMBIGUOUS_RATE="0.4"

# Parallelism across conditions (NOT within an LLM call).
# Use 1 if your provider rate-limits hard; otherwise 2–6 is usually fine.
WORKERS=4

# Cache path (keeps costs down across reruns). Must match your code expectations.
CACHE_PATH="outputs/llm_cache_openai.jsonl"

###############################################################################
# SAFETY CHECKS
###############################################################################

if [[ "${LLM_PROVIDER}" == "openai" ]]; then
  if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "Missing OPENAI_API_KEY." >&2
    exit 1
  fi
elif [[ "${LLM_PROVIDER}" == "gemini" ]]; then
  if [[ -z "${GEMINI_API_KEY:-}" && -z "${GOOGLE_API_KEY:-}" ]]; then
    echo "Missing GEMINI_API_KEY or GOOGLE_API_KEY." >&2
    exit 1
  fi
else
  echo "Unsupported LLM_PROVIDER: ${LLM_PROVIDER}. Use 'openai' or 'gemini'." >&2
  exit 1
fi

# Quick dependency check (keeps failure readable)
python - "${LLM_PROVIDER}" <<'PY'
import sys
provider = sys.argv[1] if len(sys.argv) > 1 else "openai"

if provider == "openai":
    try:
        import openai  # noqa: F401
    except Exception as e:
        raise SystemExit(
            "OpenAI SDK not importable. Install with:\n"
            "  python -m pip install -e '.[openai]'\n\n" + str(e)
        )
    print("OpenAI SDK import OK")
elif provider == "gemini":
    try:
        from google import genai  # noqa: F401
    except Exception as e:
        raise SystemExit(
            "google-genai SDK not importable. Install with:\n"
            "  python -m pip install -e '.[gemini]'  (or pip install google-genai)\n\n" + str(e)
        )
    print("Gemini SDK import OK")
else:
    raise SystemExit("Unknown provider")
PY

###############################################################################
# OUTPUT DIR + REPRODUCIBILITY
###############################################################################

RUN_TS="$(date +%Y%m%d_%H%M%S)"
OUTDIR="outputs/final_paper_run_${LLM_PROVIDER}_${RUN_TS}"
mkdir -p "${OUTDIR}"

# Record the exact settings for the paper / appendix
cat > "${OUTDIR}/command.txt" <<EOF
python -m ee_fin.cli run \\
  --outdir ${OUTDIR} \\
  --episodes ${EPISODES} \\
  --seeds ${SEEDS} \\
  --qs ${QS} \\
  --max_words ${MAX_WORDS} \\
  --senders ${SENDERS} \\
  --ambiguous_rate ${AMBIGUOUS_RATE} \\
  --workers ${WORKERS} \\
  --llm_provider ${LLM_PROVIDER} \\
  --cache_path ${CACHE_PATH} \\
  --model $( [[ "${LLM_PROVIDER}" == "openai" ]] && echo "${OPENAI_MODEL}" || echo "${GEMINI_MODEL}" )
EOF

echo "============================================================"
echo "FINAL PAPER RUN"
echo "Provider:   ${LLM_PROVIDER}"
echo "Outdir:     ${OUTDIR}"
echo "Episodes:   ${EPISODES}"
echo "Seeds:      ${SEEDS}"
echo "qs:         ${QS}"
echo "max_words:  ${MAX_WORDS}"
echo "senders:    ${SENDERS}"
echo "amb_rate:   ${AMBIGUOUS_RATE}"
echo "workers:    ${WORKERS}"
echo "cache:      ${CACHE_PATH}"
echo "============================================================"

###############################################################################
# RUN
###############################################################################

if [[ "${LLM_PROVIDER}" == "openai" ]]; then
  MODEL="${OPENAI_MODEL}"
else
  MODEL="${GEMINI_MODEL}"
fi

python -m ee_fin.cli run \
  --outdir "${OUTDIR}" \
  --episodes "${EPISODES}" \
  --seeds "${SEEDS}" \
  --qs "${QS}" \
  --max_words "${MAX_WORDS}" \
  --senders "${SENDERS}" \
  --ambiguous_rate "${AMBIGUOUS_RATE}" \
  --workers "${WORKERS}" \
  --llm_provider "${LLM_PROVIDER}" \
  --model "${MODEL}" \
  --cache_path "${CACHE_PATH}"

echo "============================================================"
echo "DONE"
echo "Wrote:"
echo "  ${OUTDIR}/logs.csv"
echo "  ${OUTDIR}/summary.csv"
echo "  ${OUTDIR}/plots/*.png + *.pdf"
echo "  ${OUTDIR}/paper_snippet.md"
echo "  ${OUTDIR}/command.txt"
echo "============================================================"
