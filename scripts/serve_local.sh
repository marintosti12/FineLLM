#!/usr/bin/env bash
# Lance l'API de triage en local (sur RunPod ou machine perso).
# Utilise le base Qwen3 + l'adapter SFT que tu as entraine.
#
# Usage:
#   HF_TOKEN=hf_xxx bash scripts/serve_local.sh
#
# L'API ecoute sur http://0.0.0.0:8000 (expose via tunnel RunPod ou ssh -L).

set -euo pipefail

# Modele de base (public HF)
export MODEL_ID="${MODEL_ID:-Qwen/Qwen3-1.7B-Base}"

# Adapter LoRA entraine (local ou HF Hub)
export ADAPTER_ID="${ADAPTER_ID:-outputs/sft/final}"

# transformers (CPU-safe) pour POC. Pour vLLM GPU: USE_VLLM=1
export USE_VLLM="${USE_VLLM:-0}"

# Port de l'API
export PORT="${PORT:-8000}"

# API key optionnelle pour demo
export API_KEY="${API_KEY:-chsa-demo-2026}"

echo "========================================"
echo "CHSA - Agent IA Triage Medical (local)"
echo "========================================"
echo "MODEL_ID  = $MODEL_ID"
echo "ADAPTER_ID= $ADAPTER_ID"
echo "USE_VLLM  = $USE_VLLM"
echo "PORT      = $PORT"
echo "API_KEY   = $API_KEY"
echo "========================================"

exec uvicorn src.deployment.serve:app \
    --host 0.0.0.0 \
    --port "$PORT" \
    --log-level info
