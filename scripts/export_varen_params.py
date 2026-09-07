#!/usr/bin/env python
"""
Export VAREN-format params JSON from 4DEquine pipeline outputs.

Produces exactly the flat schema of VAREN/examples/example_params.json:

    {"global_orient": [3], "pose": [NUM_JOINTS*3], "betas": [NUM_BETAS]}

all axis-angle, so the result can be loaded straight into VAREN / trimesh /
blender the same way the shipped example is.

USAGE:
    # first frame of the refined result -> params.json
    python scripts/export_varen_params.py \
        /home/om/mpi/4DEquine/animo-sample-out/postrefine/refined_results.pt \
        -o refined_params.json

    # one specific frame of the 174-frame raw sequence
    python scripts/export_varen_params.py \
        /home/om/mpi/4DEquine/animo-sample-out/animer_outputs.pt \
        --frame 42 -o frame042_params.json

    # every frame -> out_dir/frame_0000.json ...
    python scripts/export_varen_params.py \
        /home/om/mpi/4DEquine/animo-sample-out/animer_outputs.pt \
        --all --out-dir varen_params/
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# Key names each producer uses, in (global_orient, pose, betas) order.
KEY_SETS = [
    ('refined_global_orient', 'refined_pose', 'refined_betas'),
    ('pred_global_orient', 'pred_pose', 'pred_betas'),
    ('global_orient', 'pose', 'betas'),
]


def find_keys(data: dict):
    """Pick whichever naming convention this checkpoint uses."""
    for keys in KEY_SETS:
        if all(k in data for k in keys):
            return keys
    raise KeyError(
        f"none of the known key sets found in checkpoint. Available keys: {sorted(data.keys())}. "
        f"Expected one of: {KEY_SETS}")


def to_axis_angle(x: torch.Tensor, n_rots: int) -> torch.Tensor:
    """(...,) holding n_rots rotations as 3x3 / 6D / axis-angle -> (n_rots, 3) axis-angle.

    6D uses pytorch3d's `rotation_6d_to_matrix` because that is what 4DEquine
    itself uses to write these files (see 4DEquine/amr/models/post_optimization.py
    and eval_pose.py). Note this is the TRANSPOSE of amr.utils.geometry's own
    rot6d_to_rotmat, which reads the 6 values as two columns rather than two
    rows -- using the wrong one silently yields a plausible-looking but wrong
    pose (~0.58 mean vertex error against the stored refined mesh).
    """
    from pytorch3d.transforms import matrix_to_axis_angle, rotation_6d_to_matrix

    flat = x.reshape(-1).float()
    n = flat.numel()
    if n == n_rots * 9:
        rotmat = flat.reshape(n_rots, 3, 3)
    elif n == n_rots * 6:
        rotmat = rotation_6d_to_matrix(flat.reshape(n_rots, 6))
    elif n == n_rots * 3:
        return flat.reshape(n_rots, 3)          # already axis-angle
    else:
        raise ValueError(
            f"cannot interpret {n} values as {n_rots} rotations "
            f"(expected {n_rots*9} for rotmat, {n_rots*6} for 6D, or {n_rots*3} for axis-angle)")
    return matrix_to_axis_angle(rotmat)


def infer_num_joints(pose: torch.Tensor, requested: int) -> int:
    """Per-frame element count / (9|6|3) has to land on an integer joint count."""
    per_frame = pose[0].reshape(-1).numel()
    for width in (9, 6, 3):
        if per_frame % width == 0 and per_frame // width == requested:
            return requested
    candidates = {per_frame // w for w in (9, 6, 3) if per_frame % w == 0}
    raise ValueError(
        f"pose has {per_frame} values per frame, which is not {requested} joints in any "
        f"supported layout. Possible joint counts: {sorted(candidates)}. "
        f"Pass --num-joints with one of those.")


def frame_to_dict(data: dict, keys, idx: int, n_joints: int) -> dict:
    go_key, pose_key, betas_key = keys
    global_orient = to_axis_angle(data[go_key][idx], 1).reshape(-1)
    pose = to_axis_angle(data[pose_key][idx], n_joints).reshape(-1)
    betas = data[betas_key][idx].reshape(-1).float()
    return {
        'global_orient': global_orient.detach().cpu().tolist(),
        'pose': pose.detach().cpu().tolist(),
        'betas': betas.detach().cpu().tolist(),
    }


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="animer_outputs.pt or refined_results.pt")
    p.add_argument("-o", "--out", default="varen_params.json",
                   help="output path for a single frame (default: varen_params.json)")
    p.add_argument("--frame", type=int, default=0, help="frame index to export (default: 0)")
    p.add_argument("--all", action="store_true", help="export every frame into --out-dir")
    p.add_argument("--out-dir", default="varen_params",
                   help="directory for --all (default: varen_params/)")
    p.add_argument("--num-joints", type=int, default=37,
                   help="expected joint count in `pose` (default: 37, VAREN)")
    p.add_argument("--indent", type=int, default=None,
                   help="JSON indent; omit for the compact single-line form the example uses")
    args = p.parse_args()

    data = torch.load(args.input, map_location='cpu', weights_only=False)
    if not isinstance(data, dict):
        raise TypeError(f"expected a dict checkpoint, got {type(data).__name__}")

    keys = find_keys(data)
    go_key, pose_key, betas_key = keys
    n_frames = data[pose_key].shape[0]
    n_joints = infer_num_joints(data[pose_key], args.num_joints)
    print(f"{args.input}: {n_frames} frame(s), keys={keys}, {n_joints} joints, "
          f"{data[betas_key].shape[-1]} betas")

    if args.all:
        os.makedirs(args.out_dir, exist_ok=True)
        for i in range(n_frames):
            out = os.path.join(args.out_dir, f"frame_{i:04d}.json")
            with open(out, 'w') as f:
                json.dump(frame_to_dict(data, keys, i, n_joints), f, indent=args.indent)
        print(f"wrote {n_frames} files to {args.out_dir}/")
        return

    if not 0 <= args.frame < n_frames:
        raise IndexError(f"--frame {args.frame} out of range (file has {n_frames} frames)")
    d = frame_to_dict(data, keys, args.frame, n_joints)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(d, f, indent=args.indent)
    print(f"wrote frame {args.frame} to {args.out}  "
          f"(global_orient {len(d['global_orient'])}, pose {len(d['pose'])}, betas {len(d['betas'])})")


if __name__ == "__main__":
    main()
