#!/bin/bash
# Interactive inference with MyLLM-1B using Hugging Face Transformers.
# Usage: ./infer.sh [MODEL_DIR]
#   MODEL_DIR — path to HF model (default: ./MyLLM-1B-HF)
#
# Environment variables:
#   THREADS   — number of CPU threads (default: all available)
#   DEVICE    — torch device: cpu or cuda (default: cpu)
#   MAX_NEW   — max new tokens to generate (default: 256)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${1:-$ROOT/MyLLM-1B-HF}"
THREADS="${THREADS:-64}"
DEVICE="${DEVICE:-cpu}"
MAX_NEW="${MAX_NEW:-256}"

if [[ ! -s "$MODEL_DIR/config.json" ]]; then
  echo "Model not found at $MODEL_DIR." >&2
  echo "Make sure the safetensors weights from the GitHub Release are placed in $MODEL_DIR." >&2
  exit 1
fi

if [[ -z "$(ls "$MODEL_DIR"/*.safetensors 2>/dev/null)" ]]; then
  echo "No .safetensors weights found in $MODEL_DIR." >&2
  echo "Download them from https://github.com/gsc74/MyLLM/releases/latest and drop them in $MODEL_DIR." >&2
  exit 1
fi

# CPU performance: use all threads, compact affinity, minimize overhead
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export NUMEXPR_MAX_THREADS="$THREADS"
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=1
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto"
export TOKENIZERS_PARALLELISM=false

# Use project venv if available
if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON="$ROOT/.venv/bin/python"
elif [[ -x "$ROOT/../.venv/bin/python" ]]; then
  PYTHON="$(realpath "$ROOT/../.venv/bin/python")"
else
  PYTHON="${PYTHON:-python}"
fi

# NUMA-bind to node 0 if numactl is available
if command -v numactl &>/dev/null; then
  exec numactl --cpunodebind=0 --membind=0 "$PYTHON" "$ROOT/infer.py" "$MODEL_DIR" "$DEVICE" "$MAX_NEW"
else
  exec "$PYTHON" "$ROOT/infer.py" "$MODEL_DIR" "$DEVICE" "$MAX_NEW"
fi
