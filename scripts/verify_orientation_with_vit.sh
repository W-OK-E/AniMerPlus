#!/usr/bin/env bash
#
# verify_orientation_with_vit.sh -- overfits the real backbone+head+VAREN
# pipeline on a handful of real samples for a few hundred steps, to check the
# full learnable path converges toward the correct orientation, without
# waiting for a real ~2000-step training run. See
# scripts/verify_orientation_with_vit.py's docstring for what a PASS/FAIL
# here actually tells you (and how it complements verify_orientation_no_vit).
#
# Needs a GPU to be fast; will run on CPU but slowly.
#
# USAGE:
#   scripts/verify_orientation_with_vit.sh -j JSON -r ROOT_IMAGE [options]
#
# REQUIRED:
#   -j, --json-file PATH       horse_dataset-format JSON (e.g. .../train.json)
#   -r, --root-image PATH      ROOT_IMAGE the JSON's img_path is relative to
#
# OPTIONS:
#   -m, --varen-model-path PATH  Directory holding VAREN.pkl (default: cfg's
#                                 configured path -- usually needs overriding)
#   -n, --num-samples N          How many samples to overfit on (default: 2;
#                                 must be >=2, BatchNorm needs a real batch)
#   -s, --steps N                 Training steps (default: 300)
#   -d, --device cuda|cpu         (default: cuda if available, else cpu)
#   -h, --help                    Show this help and exit
#
# EXIT CODE: 0 if the pipeline converges (loss drop + 2D alignment), non-zero
# otherwise.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

JSON_FILE=""
ROOT_IMAGE=""
VAREN_MODEL_PATH=""
NUM_SAMPLES=2
STEPS=300
DEVICE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -j|--json-file) JSON_FILE="$2"; shift 2 ;;
    -r|--root-image) ROOT_IMAGE="$2"; shift 2 ;;
    -m|--varen-model-path) VAREN_MODEL_PATH="$2"; shift 2 ;;
    -n|--num-samples) NUM_SAMPLES="$2"; shift 2 ;;
    -s|--steps) STEPS="$2"; shift 2 ;;
    -d|--device) DEVICE="$2"; shift 2 ;;
    -h|--help) grep '^#' "$0" | sed 's/^#//; s/^ //'; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$JSON_FILE" || -z "$ROOT_IMAGE" ]]; then
  echo "Usage: $0 -j JSON -r ROOT_IMAGE [-m VAREN_MODEL_PATH] [-n NUM_SAMPLES] [-s STEPS] [-d DEVICE]" >&2
  exit 1
fi

ARGS=(--json-file "$JSON_FILE" --root-image "$ROOT_IMAGE" --num-samples "$NUM_SAMPLES" --steps "$STEPS")
[[ -n "$VAREN_MODEL_PATH" ]] && ARGS+=(--varen-model-path "$VAREN_MODEL_PATH")
[[ -n "$DEVICE" ]] && ARGS+=(--device "$DEVICE")

python3 scripts/verify_orientation_with_vit.py "${ARGS[@]}"
