#!/usr/bin/env bash
#
# validate.sh -- friendly wrapper around eval.py for evaluating a trained
# AniMerPlusPlus checkpoint.
#
# eval.py is a plain argparse script (not Hydra) that loads a *resolved*
# config snapshot (e.g. <output_dir>/.hydra/config.yaml, written automatically
# by a training run started via scripts/train.sh / main.py) and a checkpoint,
# then runs the configured Evaluator over one or more datasets.
#
# USAGE:
#   scripts/validate.sh -c CONFIG -k CHECKPOINT [options]
#
# REQUIRED:
#   -c, --config PATH          Path to a resolved Hydra config yaml, e.g.
#                                logs/train/runs/<exp_name>/.hydra/config.yaml
#   -k, --checkpoint PATH       Path to the .ckpt file to evaluate, e.g.
#                                logs/train/runs/<exp_name>/checkpoints/last.ckpt
#
# OPTIONS:
#   -s, --dataset NAME          Dataset key to evaluate, or ALL to evaluate every
#                                dataset key configured in --eval-config's DATASETS
#                                section (default: ALL). HORSE is the VAREN-shaped
#                                dataset added for this integration; it is the only
#                                dataset key this evaluator produces meaningful
#                                metrics for today (AniMerPlusPlus.forward_step is
#                                VAREN-only -- see eval.py's module docstring/comment
#                                for why the legacy ANIMAL3D/CUB/CTRLAVES3D dataset
#                                keys are not expected to produce useful numbers
#                                through it anymore).
#   -e, --eval-config PATH      Path to the (non-Hydra, plain yacs-mergeable) yaml
#                                listing dataset paths/METRIC config to evaluate
#                                against (default: amr/configs_hydra/experiment/default_val.yaml).
#                                Edit that file's DATASETS.HORSE section (or pass your
#                                own copy here) to point at real ROOT_IMAGE / JSON_FILE.TEST.
#   -d, --device DEVICE          cuda | cpu (default: cuda)
#   -h, --help                   Show this help and exit.
#
# EXAMPLES:
#   # Evaluate the HORSE (VAREN) validation split of a checkpoint from a run
#   # started with scripts/train.sh --name horse_v1:
#   scripts/validate.sh \
#       --config logs/train/runs/horse_v1/.hydra/config.yaml \
#       --checkpoint logs/train/runs/horse_v1/checkpoints/last.ckpt \
#       --dataset HORSE
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONFIG=""
CHECKPOINT=""
DATASET="ALL"
EVAL_CONFIG="${REPO_ROOT}/amr/configs_hydra/experiment/default_val.yaml"
DEVICE="cuda"

print_help() {
    sed -n '2,/^set -euo pipefail/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--config) CONFIG="$2"; shift 2 ;;
        -k|--checkpoint) CHECKPOINT="$2"; shift 2 ;;
        -s|--dataset) DATASET="$2"; shift 2 ;;
        -e|--eval-config) EVAL_CONFIG="$2"; shift 2 ;;
        -d|--device) DEVICE="$2"; shift 2 ;;
        -h|--help) print_help; exit 0 ;;
        *) echo "error: unrecognized argument '$1'" >&2; print_help; exit 1 ;;
    esac
done

if [[ -z "${CONFIG}" || -z "${CHECKPOINT}" ]]; then
    echo "error: --config and --checkpoint are required" >&2
    echo >&2
    print_help >&2
    exit 1
fi

CMD=(python "${REPO_ROOT}/eval.py"
    --config "${CONFIG}"
    --checkpoint "${CHECKPOINT}"
    --default_eval_config "${EVAL_CONFIG}"
    --dataset "${DATASET}"
    --device "${DEVICE}")

echo "+ ${CMD[*]}"
cd "${REPO_ROOT}"
exec "${CMD[@]}"
