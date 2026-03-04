#!/bin/bash
# Experiment 2: Audit budget sweep
# Tests how audit budget B (number of claims checked) affects verification effectiveness

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

echo "============================================"
echo "EXPERIMENT 2: Audit Budget Sweep"
echo "============================================"
echo ""

# Check if venv is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "⚠️  Virtual environment not activated!"
    echo "Run: source .venv/bin/activate"
    exit 1
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTDIR="outputs/exp2_audit_budget_${TIMESTAMP}"
SENDER="${1:-llm}"  # default to llm if not specified
EPISODES="${2:-100}"  # default 100 episodes
SEEDS="${3:-0,1,2,3,4}"  # default 5 seeds

# Check API key for LLM runs
if [[ "$SENDER" == "llm" ]]; then
  if [[ -z "$OPENAI_API_KEY" ]]; then
    echo "⚠️  No API key found for LLM mode!"
    echo "Set OPENAI_API_KEY"
    exit 1
  fi
fi

echo "Configuration:"
echo "  Sender: $SENDER"
echo "  Episodes: $EPISODES"
echo "  Seeds: $SEEDS"
echo "  Output: $OUTDIR"
echo ""
echo "Testing audit budgets:"
echo "  1. B=1: Check only 1 claim (intent)"
echo "  2. B=2: Check 2 claims (intent, risk_within_limit)"
echo "  3. B=4: Check all claims (full audit)"
echo ""

if [[ "$SENDER" == "llm" ]]; then
        echo "🤖 Running with OPENAI LLM (this may take a while)..."
    python -m ee_fin.cli run \
      --outdir "$OUTDIR" \
      --episodes "$EPISODES" \
      --sender llm \
            --llm_provider openai \
            --model gpt-4.1 \
      --qs 0.3,0.6,1.0 \
      --max_words 60 \
            --ambiguous_rate 0.5 \
      --approval_rewards 1.0 \
      --lie_penalties 2.0 \
      --audit_budgets 1,2,4 \
      --seeds "$SEEDS" \
    --cache_path outputs/llm_cache_openai.jsonl
else
    echo "🎯 Running with MOCK sender (fast baseline)..."
    python -m ee_fin.cli run \
      --outdir "$OUTDIR" \
      --episodes "$EPISODES" \
      --sender mock \
      --qs 0.3,0.6,1.0 \
      --max_words 60 \
    --ambiguous_rate 0.5 \
      --approval_rewards 1.0 \
      --lie_penalties 2.0 \
      --audit_budgets 1,2,4 \
      --seeds "$SEEDS"
fi

echo ""
echo "============================================"
echo "✅ EXPERIMENT 2 COMPLETE"
echo "============================================"
echo "Results saved to: $OUTDIR"
echo ""
echo "Key metrics to examine:"
echo "  - audit_fail_rate: Should vary with audit budget"
echo "  - bad_approval_rate: More checking (higher B) should reduce this"
echo "  - Compare B=1 (limited) vs B=4 (full audit)"
echo ""
echo "View summary: cat $OUTDIR/summary.csv"
echo "View plots: ls $OUTDIR/plots/"
