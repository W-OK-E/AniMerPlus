#!/usr/bin/env bash
set -euo pipefail

# Sample run for AniMerPlus with VAREN checkpoint support.
# This script assumes the AniMerPlus package root is /home/om/mpi/AniMerPlus
# and that the VAREN model files are installed under /home/om/mpi/VAREN/models.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

CHECKPOINT_PATH="${1:-${REPO_ROOT}/data/checkpoints/last.ckpt}"
IMG_FOLDER="${2:-${REPO_ROOT}/example_data}"
OUT_FOLDER="${3:-${REPO_ROOT}/demo_out}"
BATCH_SIZE="${4:-1}"
ANIMAL_TYPE="${5:-mammal}"

mkdir -p "${OUT_FOLDER}"

echo "Using checkpoint: ${CHECKPOINT_PATH}"
echo "Using VAREN model path from config: /home/om/mpi/VAREN/models"

echo "Running sample inference..."
python "${REPO_ROOT}/demo.py" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --img_folder "${IMG_FOLDER}" \
  --out_folder "${OUT_FOLDER}" \
  --batch_size "${BATCH_SIZE}" \
  --animal_type "${ANIMAL_TYPE}" \
  --save_mesh

echo "Sample run complete. Outputs written to ${OUT_FOLDER}."
