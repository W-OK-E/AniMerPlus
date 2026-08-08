#!/usr/bin/env bash
#
# verify_orientation_no_vit.sh -- fast (seconds, no GPU needed) check of the
# body-model/axis-convention plumbing, with no backbone/network involved at
# all. Feeds real ground-truth pose/shape parameters through the exact VAREN
# call + axis-fix training uses, and compares against that sample's own
# ground truth. See scripts/verify_orientation_no_vit.py's docstring for what
# a PASS/FAIL here actually tells you.
#
# USAGE:
#   scripts/verify_orientation_no_vit.sh -j JSON -r ROOT_IMAGE [options]
#
# REQUIRED:
#   -j, --json-file PATH       horse_dataset-format JSON (e.g. .../train.json)
#   -r, --root-image PATH      ROOT_IMAGE the JSON's img_path is relative to
#
# OPTIONS:
#   -m, --varen-model-path PATH  Directory holding VAREN.pkl (default: cfg's
#                                 configured path -- usually needs overriding)
#   -n, --num-samples N          How many samples to check (default: 5)
#   -o, --out PATH                Save a visual overlay PNG here (optional)
#   -h, --help                    Show this help and exit
#
# EXIT CODE: 0 if all checked samples pass, non-zero otherwise.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

JSON_FILE=""
ROOT_IMAGE=""
VAREN_MODEL_PATH=""
NUM_SAMPLES=5
OUT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -j|--json-file) JSON_FILE="$2"; shift 2 ;;
    -r|--root-image) ROOT_IMAGE="$2"; shift 2 ;;
    -m|--varen-model-path) VAREN_MODEL_PATH="$2"; shift 2 ;;
    -n|--num-samples) NUM_SAMPLES="$2"; shift 2 ;;
    -o|--out) OUT="$2"; shift 2 ;;
    -h|--help) grep '^#' "$0" | sed 's/^#//; s/^ //'; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$JSON_FILE" || -z "$ROOT_IMAGE" ]]; then
  echo "Usage: $0 -j JSON -r ROOT_IMAGE [-m VAREN_MODEL_PATH] [-n NUM_SAMPLES] [-o OUT.png]" >&2
  exit 1
fi

ARGS=(--json-file "$JSON_FILE" --root-image "$ROOT_IMAGE" --num-samples "$NUM_SAMPLES")
[[ -n "$VAREN_MODEL_PATH" ]] && ARGS+=(--varen-model-path "$VAREN_MODEL_PATH")
[[ -n "$OUT" ]] && ARGS+=(--out "$OUT")

python3 scripts/verify_orientation_no_vit.py "${ARGS[@]}"
