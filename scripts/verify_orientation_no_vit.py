#!/usr/bin/env python
"""
Orientation sanity check WITHOUT the ViT backbone / learned head.

Takes real ground-truth pose/shape parameters straight from a horse dataset
JSON (not network predictions) and runs them through the exact same VAREN
call + axis-fix that forward_one_parametric_model (amr/models/animerpp.py)
uses, then compares the result to that same sample's own exported
ground-truth keypoints, point-0-relative (matches Keypoint3DLoss's own
normalization).

This isolates the body-model + axis-convention plumbing from anything the
network has or hasn't learned. It runs in seconds and needs no GPU, no
backbone weights, and no training -- use it before waiting ~2000 training
steps to find out whether the output looks inverted.

  PASS here  = the body-model/axis plumbing is correct. If training still
               looks wrong after many steps, look at what the network is
               being asked to *learn* (e.g. the rotation representation),
               not this plumbing -- see verify_orientation_with_vit.py.
  FAIL here  = the bug is in this plumbing, independent of training at all.

Usage:
    python scripts/verify_orientation_no_vit.py \
        --json-file /path/to/horse_dataset/train.json \
        --root-image /path/to/root_image_dir \
        [--num-samples 5] [--out orientation_check_no_vit.png]

Exits non-zero if any checked sample's mean keypoint error exceeds
--error-threshold.
"""
import argparse
import json
import os
import sys
import traceback

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

import numpy as np
import torch

from amr.configs import get_config
from amr.models.animerpp import _varen_native_to_camera_frame
from amr.models.varen_warapper import VAREN
from amr.utils.geometry import perspective_projection

# The dataset's exported frame (+Y up) and the pipeline's camera frame (+Y down,
# +Z depth) differ by a 180-degree rotation about X -- negating Y and Z converts
# either way. See .agent/Checks.md (2026-08-10, N1-N4).
EXPORT_FROM_CAMERA = np.array([1., -1., -1.], dtype=np.float32)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="amr/configs_hydra/experiment/AniMerPlus.yaml")
    p.add_argument("--json-file", required=True, help="Path to a horse_dataset-format JSON (e.g. train.json)")
    p.add_argument("--root-image", required=True, help="ROOT_IMAGE the JSON's img_path/mask_path are relative to")
    p.add_argument("--varen-model-path", default=None,
                  help="Override cfg.VAREN.MODEL_PATH (the directory holding VAREN.pkl) -- "
                       "the config's default is a cluster path, so this almost always needs "
                       "setting explicitly unless you're already running where that path resolves.")
    p.add_argument("--num-samples", type=int, default=5)
    p.add_argument("--error-threshold", type=float, default=0.05,
                  help="Max allowed mean per-point 3D error (model units, same units the "
                       "training loss uses) before a sample is reported FAIL")
    p.add_argument("--reproj-threshold-px", type=float, default=15.0,
                  help="Max allowed best-case 2D reprojection error, in px of a 256px crop. "
                       "Catches coordinate-frame errors that the 3D check is blind to (a "
                       "frame flip lands around ~104px here); correct geometry sits at ~2-3px, "
                       "the residual being the dataset's weak-perspective approximation.")
    p.add_argument("--out", default=None,
                  help="Optional: save a visual overlay (real image + our mesh's silhouette) here. "
                       "Best-effort -- skipped with a warning if the real image/camera params "
                       "aren't reachable from --root-image; the numeric check above is the "
                       "authoritative pass/fail signal either way.")
    return p.parse_args()


def _best_reprojection_error(kp3d: np.ndarray, kp2d_raw: np.ndarray, bbox, image_size: int) -> float:
    """Best achievable reprojection error (px) of kp3d onto this sample's own GT
    2D keypoints, over all camera translations.

    Solves only for the 3-number camera translation -- deliberately nothing else,
    so the result reflects the *geometry* (and therefore the coordinate frame) and
    can't be fudged by fitting extra parameters. Reported in pixels of an
    image_size-square crop of the sample's bbox, matching what training sees.
    """
    kp2d = kp2d_raw[:, :2].copy()
    conf = kp2d_raw[:, 2] > 0 if kp2d_raw.shape[1] > 2 else np.ones(len(kp2d), dtype=bool)
    if bbox is not None:  # express GT in the same normalized crop coords training uses
        cx, cy = bbox[0] + bbox[2] / 2.0, bbox[1] + bbox[3] / 2.0
        size = max(bbox[2], bbox[3])
        kp2d = (kp2d - np.array([cx, cy])) / size
    else:
        kp2d = kp2d / image_size - 0.5

    P = torch.tensor(kp3d, dtype=torch.float64).unsqueeze(0)
    target = torch.tensor(kp2d, dtype=torch.float64)
    mask = torch.tensor(conf)
    focal = torch.tensor([[1000.0, 1000.0]], dtype=torch.float64) / image_size
    t = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        t[2] = 6.0
    opt = torch.optim.Adam([t], lr=0.05)
    for _ in range(1200):
        opt.zero_grad()
        proj = perspective_projection(P, translation=t.unsqueeze(0), focal_length=focal)
        ((proj[0][mask] - target[mask]) ** 2).sum(-1).mean().backward()
        opt.step()
    with torch.no_grad():
        proj = perspective_projection(P, translation=t.unsqueeze(0), focal_length=focal)
        return float((proj[0][mask] - target[mask]).norm(dim=-1).mean() * image_size)


def try_render_overlay(samples, verts_all, faces, root_image, out_path):
    """Best-effort visual check: overlay our mesh's silhouette on the real
    training image. Not required for the pass/fail verdict -- only run if
    matplotlib/cv2 and the actual images/camera params are available."""
    try:
        import cv2
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from PIL import Image
    except ImportError as exc:
        print(f"[visual check skipped] missing dependency: {exc}")
        return

    fig, axes = plt.subplots(1, len(samples), figsize=(7 * len(samples), 7))
    axes = [axes] if len(samples) == 1 else list(axes)
    any_rendered = False

    for s, verts, ax in zip(samples, verts_all, axes):
        img_path = os.path.join(root_image, s['img_path'])
        if not os.path.exists(img_path):
            ax.text(0.5, 0.5, "image not found\n(this is fine -- see numeric check above)",
                   ha='center', va='center', wrap=True)
            ax.axis('off')
            continue

        img = np.array(Image.open(img_path).convert('RGB'))
        trans = np.array(s['trans'], dtype=np.float32)

        # exact camera params genzoo used, if reachable (batch_dir/control/name_params.json) --
        # gives a pixel-exact overlay. Falls back to an approximate true-perspective
        # projection (see .agent/Checks.md's weak-vs-real-perspective note) otherwise.
        name = os.path.splitext(os.path.basename(s['img_path']))[0]
        batch_dir = os.path.dirname(os.path.dirname(img_path))
        control_path = os.path.join(batch_dir, 'control', f'{name}_params.json')
        if os.path.exists(control_path):
            sys.path.insert(0, os.path.dirname(control_path))  # harmless if already skipped
            try:
                genzoo_dir = None
                for candidate in (os.environ.get('GENZOO_DIR'), '/home/om/mpi/data-gen/genzoo'):
                    if candidate and os.path.isdir(candidate):
                        genzoo_dir = candidate
                        break
                if genzoo_dir:
                    sys.path.insert(0, genzoo_dir)
                from eval_dummy_pipeline import project_analytic
                with open(control_path) as f:
                    camera_params = np.array(json.load(f)['camera_params'], dtype=np.float32)
                proj = project_analytic(verts, camera_params, img.shape[0])
            except Exception:
                proj = None
        else:
            proj = None

        if proj is None:
            from amr.utils.geometry import perspective_projection
            v = torch.tensor(verts, dtype=torch.float32).unsqueeze(0)
            t = torch.tensor(trans, dtype=torch.float32).unsqueeze(0)
            f = torch.tensor([[1000.0, 1000.0]], dtype=torch.float32)
            c = torch.tensor([[img.shape[1] / 2, img.shape[0] / 2]], dtype=torch.float32)
            proj = perspective_projection(v, translation=t, focal_length=f, camera_center=c)[0].numpy()

        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, proj[faces].astype(np.int32), 255)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        overlay = img.copy()
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 4)
        ax.imshow(overlay)
        ax.axis('off')
        any_rendered = True

    if any_rendered:
        plt.tight_layout()
        plt.savefig(out_path, dpi=110)
        print(f"Saved visual overlay to {out_path}")
    else:
        print("[visual check skipped] none of the sampled images were reachable from --root-image")


def main():
    args = parse_args()
    cfg = get_config(args.config)
    cfg.defrost()
    if args.varen_model_path:
        cfg.VAREN.MODEL_PATH = args.varen_model_path
    cfg.freeze()

    with open(args.json_file) as f:
        all_data = json.load(f)['data']
    samples = all_data[:args.num_samples]
    print(f"Checking {len(samples)} sample(s) from {args.json_file}")

    varen = VAREN(model_path=cfg.VAREN.MODEL_PATH, num_betas=cfg.VAREN.get('NUM_BETAS', 39),
                  use_muscle_deformations=cfg.VAREN.get('USE_MUSCLE_DEFORMATIONS', False), ext='pkl')
    faces = varen.faces

    all_pass = True
    verts_for_render = []
    for i, s in enumerate(samples):
        pose = np.array(s['pose'], dtype=np.float32)
        betas = np.array(s['shape'], dtype=np.float32)

        global_orient = torch.tensor(pose[:3]).unsqueeze(0)
        body_pose = torch.tensor(pose[3:]).unsqueeze(0)
        betas_t = torch.tensor(betas).unsqueeze(0)
        out = varen(global_orient=global_orient, body_pose=body_pose, betas=betas_t,
                    transl=None, pose2rot=True)

        verts_fixed = _varen_native_to_camera_frame(out.vertices)[0].detach().numpy()
        kp_fixed = _varen_native_to_camera_frame(out.surface_keypoints)[0].detach().numpy()
        verts_for_render.append(verts_fixed * EXPORT_FROM_CAMERA)
        kp_gt = np.array(s['keypoint_3d'], dtype=np.float32) * EXPORT_FROM_CAMERA
        err = np.linalg.norm((kp_fixed - kp_fixed[0]) - (kp_gt - kp_gt[0]), axis=-1)
        mean_err = float(err.mean())
        status = "PASS" if mean_err < args.error_threshold else "FAIL"
        if status == "FAIL":
            all_pass = False
        px_err = _best_reprojection_error(kp_fixed, np.array(s['keypoint_2d'], dtype=np.float32),
                                          s.get('bbox'), cfg.MODEL.IMAGE_SIZE)
        px_status = "PASS" if px_err < args.reproj_threshold_px else "FAIL"
        if px_status == "FAIL":
            all_pass = False
        print(f"  sample {i:3d} ({s['img_path']}): mean 3D keypoint error = {mean_err:.4f}  [{status}]"
             f"   |  best 2D reprojection error = {px_err:6.2f}px  [{px_status}]")

    if args.out:
        try_render_overlay(samples, verts_for_render, faces, args.root_image, args.out)

    print()
    if all_pass:
        print(f"PASS -- body-model/axis plumbing matches the dataset's ground truth "
             f"(all samples under {args.error_threshold} error).")
    else:
        print(f"FAIL -- at least one sample exceeded {args.error_threshold} error. "
             f"The plumbing (VAREN forward + axis remap in animerpp.py) does not match "
             f"this dataset's ground truth -- check that _varen_native_to_camera_frame "
             f"is actually present and unchanged, and that --varen-model-path points at "
             f"the VAREN copy with the check_inputs/keypoint-ordering fixes.")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\nVERIFY-ORIENTATION (no ViT) SCRIPT FAILED WITH AN EXCEPTION:", file=sys.stderr)
        traceback.print_exc()
        sys.exit(2)
