#!/bin/bash
# Quick sanity check script - runs 2 episodes to test setup

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

echo "============================================"
echo "SANITY CHECK: Running 2 episodes"
echo "============================================"

# Check if venv is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "⚠️  Virtual environment not activated!"
    echo "Run: source .venv/bin/activate"
    exit 1
fi

# Check API key
if [[ -z "$GEMINI_API_KEY" ]] && [[ -z "$GOOGLE_API_KEY" ]]; then
    echo "⚠️  No API key found!"
    echo "Set GEMINI_API_KEY or GOOGLE_API_KEY"
    exit 1
fi

echo "✓ Virtual environment: $VIRTUAL_ENV"
echo "✓ API key: ${GEMINI_API_KEY:0:10}..."
echo ""

# Test with mock sender first
echo "1️⃣  Testing with MOCK sender (no API calls)..."
python -m ee_fin.cli run \
  --outdir outputs/sanity_mock \
  --episodes 2 \
  --qs 0,1.0 \
  --max_words 60 \
  --sender mock

echo ""
echo "✅ Mock sender test passed!"
echo ""

# Test with Gemini
echo "2️⃣  Testing with GEMINI LLM (real API calls)..."
python -m ee_fin.cli run \
  --outdir outputs/sanity_gemini_$(date +%H%M%S) \
  --episodes 2 \
  --qs 0,1.0 \
  --max_words 60 \
  --sender llm \
  --llm_provider gemini \
  --debug

echo ""
echo "============================================"
echo "✅ SANITY CHECK COMPLETE"
echo "============================================"
echo "Check outputs/sanity_* for results"
