#!/usr/bin/env bash
#
# test_camera_scale_overfit.sh -- overfit-a-tiny-batch check for the
# camera-scale fix applied to amr/models/animerpp.py on 2026-08-25 (predicted
# camera scale now goes through softplus so it can't cross zero / blow up
# unboundedly as a divisor). See scripts/test_camera_scale_overfit.py's
# docstring for full context and what each field in the output means.
#
# Unlike verify_orientation_with_vit.sh, this also applies the backbone-freeze
# overrides real training uses (run.sh's FREEZE_ATTN/FREEZE_FFN/USE_CLS) --
# without them the full ViT-MoE backbone is trainable and OOMs on a 12GB card.
#
# Runs in the animer2 micromamba env automatically.
#
# USAGE:
#   scripts/test_camera_scale_overfit.sh [options]
#
# OPTIONS (all have defaults matching the 2026-08-25 debugging session):
#   -j, --json-file PATH        (default: animer_train_data/horse_dataset_textured/train.json)
#   -r, --root-image PATH       (default: animer_train_data/batches)
#   -m, --varen-model-path PATH (default: /home/om/mpi/VAREN/models)
#   -w, --pretrained-weights PATH  Real backbone checkpoint (default:
#                                data/AniMerPlus/checkpoint.ckpt). Pass "" for
#                                a fast random-init smoke check instead.
#   -n, --num-samples N         (default: 10)
#   --num-holdout-samples N      Samples NOT used for training, for the final
#                                results/render (default: 10)
#   -s, --steps N                (default: 800)
#   --seed N                     torch.manual_seed, for reproducibility (default: unset)
#   --sample-offset F            fraction-of-bucket sample-selection offset,
#                                0.0 or 0.5 were used to get two different
#                                sample sets in the original session (default: 0.0)
#   -d, --device cuda|cpu        (default: cuda if available)
#   --disable-fix                Use the OLD unconstrained formula instead (for
#                                a side-by-side comparison against the fix)
#   -o, --render-out PATH        (default: camera_scale_overfit_render.png)
#   --no-render                  Numeric results only, skip the image
#   --checkpoint-dir PATH        Periodic checkpoints + per-step render snapshots
#                                (default: camera_scale_overfit_checkpoints)
#   --checkpoint-every N          Steps between checkpoints (default: same as
#                                the --log-every the .py script uses, 50). 0 disables.
#   --resume-from PATH            Resume from a step_NNNNNN.pt checkpoint. --steps
#                                is the TARGET total, not additional steps. Must
#                                use the same sample/model config it was saved with.
#   -h, --help                   Show this help and exit
#
# EXAMPLES:
#   # Reproduce the passing "with fix" run from the debugging session:
#   scripts/test_camera_scale_overfit.sh
#
#   # Reproduce the pre-fix baseline for comparison:
#   scripts/test_camera_scale_overfit.sh --disable-fix -o baseline_render.png
#
#   # Same samples/seed as the second ("does it reproduce") confirmation run:
#   scripts/test_camera_scale_overfit.sh --seed 1234 --sample-offset 0.5
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

JSON_FILE="/home/om/mpi/animer_train_data/horse_dataset_textured/train.json"
ROOT_IMAGE="/home/om/mpi/animer_train_data/batches"
VAREN_MODEL_PATH="/home/om/mpi/VAREN/models"
PRETRAINED_WEIGHTS="data/AniMerPlus/checkpoint.ckpt"
NUM_SAMPLES=10
NUM_HOLDOUT_SAMPLES=10
STEPS=800
SEED=""
SAMPLE_OFFSET="0.0"
DEVICE=""
DISABLE_FIX=0
RENDER_OUT=""
NO_RENDER=0
CHECKPOINT_DIR=""
CHECKPOINT_EVERY=""
RESUME_FROM=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -j|--json-file) JSON_FILE="$2"; shift 2 ;;
    -r|--root-image) ROOT_IMAGE="$2"; shift 2 ;;
    -m|--varen-model-path) VAREN_MODEL_PATH="$2"; shift 2 ;;
    -w|--pretrained-weights) PRETRAINED_WEIGHTS="$2"; shift 2 ;;
    -n|--num-samples) NUM_SAMPLES="$2"; shift 2 ;;
    --num-holdout-samples) NUM_HOLDOUT_SAMPLES="$2"; shift 2 ;;
    -s|--steps) STEPS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --sample-offset) SAMPLE_OFFSET="$2"; shift 2 ;;
    -d|--device) DEVICE="$2"; shift 2 ;;
    --disable-fix) DISABLE_FIX=1; shift ;;
    -o|--render-out) RENDER_OUT="$2"; shift 2 ;;
    --no-render) NO_RENDER=1; shift ;;
    --checkpoint-dir) CHECKPOINT_DIR="$2"; shift 2 ;;
    --checkpoint-every) CHECKPOINT_EVERY="$2"; shift 2 ;;
    --resume-from) RESUME_FROM="$2"; shift 2 ;;
    -h|--help) grep '^#' "$0" | sed 's/^#//; s/^ //'; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

ARGS=(--json-file "$JSON_FILE" --root-image "$ROOT_IMAGE" --varen-model-path "$VAREN_MODEL_PATH"
      --pretrained-weights "$PRETRAINED_WEIGHTS" --num-samples "$NUM_SAMPLES"
      --num-holdout-samples "$NUM_HOLDOUT_SAMPLES" --steps "$STEPS"
      --sample-offset "$SAMPLE_OFFSET")
[[ -n "$SEED" ]] && ARGS+=(--seed "$SEED")
[[ -n "$DEVICE" ]] && ARGS+=(--device "$DEVICE")
[[ "$DISABLE_FIX" -eq 1 ]] && ARGS+=(--disable-fix)
[[ -n "$RENDER_OUT" ]] && ARGS+=(--render-out "$RENDER_OUT")
[[ "$NO_RENDER" -eq 1 ]] && ARGS+=(--no-render)
[[ -n "$CHECKPOINT_DIR" ]] && ARGS+=(--checkpoint-dir "$CHECKPOINT_DIR")
[[ -n "$CHECKPOINT_EVERY" ]] && ARGS+=(--checkpoint-every "$CHECKPOINT_EVERY")
[[ -n "$RESUME_FROM" ]] && ARGS+=(--resume-from "$RESUME_FROM")

micromamba run -n animer2 python3 scripts/test_camera_scale_overfit.py "${ARGS[@]}"
