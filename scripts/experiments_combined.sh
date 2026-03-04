#!/bin/bash
# Combined experiments: Run both Experiment 1 and Experiment 2

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

echo "============================================"
echo "COMBINED EXPERIMENTS: Adversarial Incentives + Audit Budget"
echo "============================================"
echo ""

# Check if venv is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "⚠️  Virtual environment not activated!"
    echo "Run: source .venv/bin/activate"
    exit 1
fi

# Check API key for LLM runs
if [[ "$1" == "llm" ]]; then
    if [[ -z "${OPENAI_API_KEY:-}" ]]; then
        echo "⚠️  No API key found for LLM mode!"
        echo "Set OPENAI_API_KEY"
        exit 1
    fi
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTDIR="outputs/combined_experiments_${TIMESTAMP}"
SENDER="${1:-llm}"  # default to llm if not specified
EPISODES="${2:-100}"  # default 100 episodes

echo "Configuration:"
echo "  Sender: $SENDER"
echo "  Episodes: $EPISODES"
echo "  Output: $OUTDIR"
echo ""
echo "Testing full factorial design:"
echo "  V (approval_reward): 1.0, 2.0"
echo "  L (lie_penalty): 1.0, 2.0, 4.0"
echo "  B (audit_budget): 1, 2, 4, None"
echo "  q (audit_prob): 0.3, 0.6, 1.0"
echo ""
echo "This will run many conditions and may take a while..."
echo ""

if [[ "$SENDER" == "llm" ]]; then
        echo "🤖 Running with OpenAI LLM (canonical default)..."
    python -m ee_fin.cli run \
      --outdir "$OUTDIR" \
      --episodes "$EPISODES" \
      --sender llm \
            --llm_provider openai \
            --model gpt-4.1 \
      --qs 0.3,0.6,1.0 \
      --max_words 60 \
      --approval_rewards 1.0,2.0 \
      --lie_penalties 1.0,2.0,4.0 \
      --audit_budgets 1,2,4,None \
    --cache_path outputs/llm_cache_openai.jsonl \
      --workers 1
else
    echo "🎯 Running with MOCK sender..."
    python -m ee_fin.cli run \
      --outdir "$OUTDIR" \
      --episodes "$EPISODES" \
      --sender mock \
      --qs 0.3,0.6,1.0 \
      --max_words 60 \
      --approval_rewards 1.0,2.0 \
      --lie_penalties 1.0,2.0,4.0 \
      --audit_budgets 1,2,4,None \
      --workers 1
fi

echo ""
echo "============================================"
echo "✅ COMBINED EXPERIMENTS COMPLETE"
echo "============================================"
echo "Results saved to: $OUTDIR"
echo ""
echo "Analysis suggestions:"
echo "  1. Filter by V, L to see incentive effects (Experiment 1)"
echo "  2. Filter by B to see audit budget effects (Experiment 2)"
echo "  3. Look for interactions between V, L, and B"
echo ""
echo "View summary: cat $OUTDIR/summary.csv"
echo "View logs: cat $OUTDIR/logs.csv"
