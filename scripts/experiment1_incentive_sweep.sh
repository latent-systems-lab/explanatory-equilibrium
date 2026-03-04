#!/bin/bash
# Experiment 1: Adversarial incentive sweep
# Tests how V (approval_reward) and L (lie_penalty) affect lying behavior

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

echo "============================================"
echo "EXPERIMENT 1: Adversarial Incentive Sweep"
echo "============================================"
echo ""

# Check if venv is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "⚠️  Virtual environment not activated!"
    echo "Run: source .venv/bin/activate"
    exit 1
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTDIR="outputs/exp1_incentive_sweep_${TIMESTAMP}"
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
echo "Testing exact 3 conditions:"
echo "  1. BASELINE:    V=1, L=2"
echo "  2. TEMPTATION:  V=2, L=1"
echo "  3. PUNISHMENT:  V=1, L=4"
echo ""

if [[ "$SENDER" == "llm" ]]; then
        echo "🤖 Running BASELINE (V=1, L=2)"
        python -m ee_fin.cli run \
            --outdir "$OUTDIR/baseline_V1_L2" \
            --episodes "$EPISODES" \
            --sender llm \
            --llm_provider openai \
            --model gpt-4.1 \
            --qs 0.3,0.6,1.0 \
            --max_words 60 \
            --ambiguous_rate 0.5 \
            --approval_rewards 1.0 \
            --lie_penalties 2.0 \
            --audit_budgets None \
            --seeds "$SEEDS" \
            --cache_path outputs/llm_cache_openai.jsonl

        echo "🤖 Running TEMPTATION (V=2, L=1)"
        python -m ee_fin.cli run \
            --outdir "$OUTDIR/temptation_V2_L1" \
            --episodes "$EPISODES" \
            --sender llm \
            --llm_provider openai \
            --model gpt-4.1 \
            --qs 0.3,0.6,1.0 \
            --max_words 60 \
            --ambiguous_rate 0.5 \
            --approval_rewards 2.0 \
            --lie_penalties 1.0 \
            --audit_budgets None \
            --seeds "$SEEDS" \
            --cache_path outputs/llm_cache_openai.jsonl

        echo "🤖 Running PUNISHMENT (V=1, L=4)"
        python -m ee_fin.cli run \
            --outdir "$OUTDIR/punishment_V1_L4" \
            --episodes "$EPISODES" \
            --sender llm \
            --llm_provider openai \
            --model gpt-4.1 \
            --qs 0.3,0.6,1.0 \
            --max_words 60 \
            --ambiguous_rate 0.5 \
            --approval_rewards 1.0 \
            --lie_penalties 4.0 \
            --audit_budgets None \
            --seeds "$SEEDS" \
            --cache_path outputs/llm_cache_openai.jsonl
else
        echo "🎯 Running BASELINE (V=1, L=2)"
        python -m ee_fin.cli run \
            --outdir "$OUTDIR/baseline_V1_L2" \
            --episodes "$EPISODES" \
            --sender mock \
            --qs 0.3,0.6,1.0 \
            --max_words 60 \
            --ambiguous_rate 0.5 \
            --approval_rewards 1.0 \
            --lie_penalties 2.0 \
            --audit_budgets None \
            --seeds "$SEEDS"

        echo "🎯 Running TEMPTATION (V=2, L=1)"
        python -m ee_fin.cli run \
            --outdir "$OUTDIR/temptation_V2_L1" \
            --episodes "$EPISODES" \
            --sender mock \
            --qs 0.3,0.6,1.0 \
            --max_words 60 \
            --ambiguous_rate 0.5 \
            --approval_rewards 2.0 \
            --lie_penalties 1.0 \
            --audit_budgets None \
            --seeds "$SEEDS"

        echo "🎯 Running PUNISHMENT (V=1, L=4)"
        python -m ee_fin.cli run \
            --outdir "$OUTDIR/punishment_V1_L4" \
            --episodes "$EPISODES" \
            --sender mock \
            --qs 0.3,0.6,1.0 \
            --max_words 60 \
            --ambiguous_rate 0.5 \
            --approval_rewards 1.0 \
            --lie_penalties 4.0 \
            --audit_budgets None \
            --seeds "$SEEDS"
fi

echo ""
echo "============================================"
echo "✅ EXPERIMENT 1 COMPLETE"
echo "============================================"
echo "Results saved to: $OUTDIR"
echo ""
echo "Key metrics to examine:"
echo "  - bad_approval_rate: Should increase with V=2,L=1 (temptation)"
echo "  - audit_fail_rate: Shows lying caught by audits"
echo "  - avg_welfare: Impact on overall welfare"
echo ""
echo "View summaries:"
echo "  cat $OUTDIR/baseline_V1_L2/summary.csv"
echo "  cat $OUTDIR/temptation_V2_L1/summary.csv"
echo "  cat $OUTDIR/punishment_V1_L4/summary.csv"
