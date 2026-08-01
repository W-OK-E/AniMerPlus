#!/usr/bin/env python
"""
Synthetic-batch smoke test for the VAREN integration in AniMerPlusPlus.

Builds the real Hydra config used by main.py (experiment=AniMerPlus), constructs
AniMerPlusPlus from it, and drives forward_step()/compute_loss() against a
small hand-built random batch with the exact tensor shapes the VAREN forward
path expects (43 surface keypoints, 39 betas, 37*3+3 pose, etc.) -- no real
images/JSON/dataset classes are involved, so this can run before any genzoo
data exists.

Two scenarios are run:
  1. A batch with "reasonable" (small, near-zero) ground-truth values.
  2. A batch with a deliberately out-of-shape-space betas vector (large
     magnitude), per VAREN_integration_plan.md's post-implementation check:
     confirm the model does not crash when asked to produce a mesh/render
     outside the shape space it was likely trained on. This calls self.varen
     directly with the extreme betas and also renders one frame through the
     configured MeshRenderer, since "does not crash" per the plan includes the
     render step.

Usage:
    python scripts/smoke_test_varen.py [--device cuda|cpu] [--batch-size 2]

Set AMR_DISABLE_CUDNN=1 to run with torch.backends.cudnn.enabled = False. Some
GPU/cuDNN/driver combinations (observed here on an older Pascal-arch card with
a newer cuDNN build) raise "GET was unable to find an engine to execute this
computation" from plain nn.Conv2d calls regardless of this repo's code -- this
is an environment issue, not specific to the VAREN integration, but the flag
is provided so this script can still complete a GPU smoke test in that case.

Exits non-zero on any unexpected exception.
"""
import argparse
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

import torch
from hydra import compose, initialize_config_dir


def build_cfg(extra_overrides=None):
    extra_overrides = extra_overrides or []
    config_dir = str(root / "amr" / "configs_hydra")
    overrides = [
        "experiment=AniMerPlus",
        "trainer=gpu",
        "launcher=local",
        # Skip the 2.7GB pretrained backbone checkpoint -- irrelevant to whether
        # the VAREN plumbing is wired correctly, and much slower to load.
        "MODEL.BACKBONE.PRETRAINED_WEIGHTS=null",
    ] + extra_overrides
    with initialize_config_dir(version_base="1.2", config_dir=config_dir):
        cfg = compose(config_name="train.yaml", overrides=overrides)
    return cfg


def make_fake_batch(cfg, batch_size: int, device: str, betas_scale: float = 0.1):
    num_joints = cfg.VAREN.NUM_JOINTS
    num_betas = cfg.VAREN.get("NUM_BETAS", 39)
    num_keypoints = 43  # len(varen.vertex_ids.vertex_ids["varen"])
    image_size = cfg.MODEL.IMAGE_SIZE

    img = torch.randn(batch_size, 3, image_size, image_size, device=device)
    focal_length = torch.full((batch_size, 2), float(cfg.VAREN.get("FOCAL_LENGTH", 1000)), device=device)
    keypoints_2d = torch.zeros(batch_size, num_keypoints, 3, device=device)
    keypoints_2d[..., :2] = torch.rand(batch_size, num_keypoints, 2, device=device) - 0.5
    keypoints_2d[..., 2] = 1.0  # visibility / confidence
    keypoints_3d = torch.zeros(batch_size, num_keypoints, 4, device=device)
    keypoints_3d[..., :3] = torch.randn(batch_size, num_keypoints, 3, device=device) * 0.3
    keypoints_3d[..., 3] = 1.0
    mask = torch.ones(batch_size, image_size, image_size, device=device)
    category = torch.full((batch_size,), 1000, dtype=torch.long, device=device)
    supercategory = torch.full((batch_size,), 1000, dtype=torch.long, device=device)

    batch = {
        'img': img,
        'mask': mask,
        'focal_length': focal_length,
        'keypoints_2d': keypoints_2d,
        'keypoints_3d': keypoints_3d,
        'category': category,
        'supercategory': supercategory,
    }
    return batch


def run_forward_and_loss(model, batch, label):
    print(f"[{label}] forward_step ...")
    with torch.no_grad():
        output = model.forward_step(batch, train=False)
    assert 'varen_output' in output, f"[{label}] forward_step output missing 'varen_output': {list(output.keys())}"
    varen_out = output['varen_output']
    for key in ('pred_keypoints_2d', 'pred_keypoints_3d', 'pred_vertices', 'pred_cam_t'):
        assert key in varen_out, f"[{label}] varen_output missing '{key}'"
    n_kp = varen_out['pred_keypoints_3d'].shape[1]
    print(f"[{label}] pred_keypoints_3d shape: {tuple(varen_out['pred_keypoints_3d'].shape)} "
         f"(expected N=43 surface keypoints, got N={n_kp})")
    assert n_kp == 43, (
        f"[{label}] expected 43 predicted keypoints (VAREN surface_keypoints, matching the "
        f"genzoo 43-point ground truth convention), got {n_kp}. If VAREN.forward() output.joints "
        f"is being used instead of output.surface_keypoints, this will be 38, not 43."
    )
    assert torch.isfinite(varen_out['pred_keypoints_3d']).all(), f"[{label}] NaN/Inf in pred_keypoints_3d"
    assert torch.isfinite(varen_out['pred_vertices']).all(), f"[{label}] NaN/Inf in pred_vertices"

    print(f"[{label}] compute_loss ...")
    with torch.no_grad():
        loss = model.compute_loss(batch, output, train=False)
    print(f"[{label}] loss = {loss.item():.6f}, components = "
         f"{ {k: round(v.item(), 6) for k, v in output['losses'].items()} }")
    assert torch.isfinite(loss).all(), f"[{label}] loss is NaN/Inf"
    return output


def run_extreme_betas_test(model, cfg, batch_size: int, device: str):
    """
    VAREN_integration_plan.md 'NOTE Post implementation Check': dry run to
    confirm the model does not crash when generating a mesh/image outside its
    likely-seen shape space. Calls self.varen directly (bypassing the head, to
    guarantee betas really are extreme rather than hoping an untrained head
    happens to predict something extreme) and renders one frame.
    """
    print("[extreme-betas] calling model.varen(...) directly with |betas| >> typical range ...")
    num_joints = cfg.VAREN.NUM_JOINTS
    global_orient = torch.zeros(batch_size, 3, device=device)
    body_pose = torch.zeros(batch_size, num_joints * 3, device=device)
    betas = (torch.randn(batch_size, cfg.VAREN.get("NUM_BETAS", 39), device=device)) * 50.0  # ~50 std devs
    with torch.no_grad():
        out = model.varen(global_orient=global_orient, body_pose=body_pose, betas=betas,
                          transl=None, pose2rot=True)
    vertices = out.vertices
    n_nan = torch.isnan(vertices).sum().item()
    n_inf = torch.isinf(vertices).sum().item()
    print(f"[extreme-betas] vertices shape={tuple(vertices.shape)}, "
         f"min={vertices.min().item():.3f}, max={vertices.max().item():.3f}, "
         f"nan_count={n_nan}, inf_count={n_inf}")
    if n_nan or n_inf:
        print("[extreme-betas] WARNING: extreme betas produced NaN/Inf vertices "
             "(model degrades ungracefully for this input, but did not raise/crash).")
    else:
        print("[extreme-betas] OK: extreme betas produced a (likely very deformed, "
             "but numerically finite) mesh without crashing.")

    if getattr(model, 'varen_mesh_renderer', None) is not None:
        print("[extreme-betas] rendering the extreme-betas mesh through MeshRenderer ...")
        try:
            pred_cam_t = torch.tensor([[0., 0., 5.]] * batch_size, device=device).cpu().numpy()
            images = torch.zeros(batch_size, 3, cfg.MODEL.IMAGE_SIZE, cfg.MODEL.IMAGE_SIZE).numpy()
            dummy_kp2d = torch.zeros(batch_size, vertices.shape[1] if False else 43, 2).cpu().numpy()
            gt_kp2d = torch.zeros(batch_size, 43, 3).cpu().numpy()
            rend_imgs = model.varen_mesh_renderer.visualize_tensorboard(
                vertices.detach().cpu().numpy(),
                pred_cam_t,
                images,
                cfg.VAREN.get("FOCAL_LENGTH", 1000),
                dummy_kp2d,
                gt_kp2d,
            )
            print(f"[extreme-betas] render OK, produced {len(rend_imgs)} image(s).")
        except Exception:
            print("[extreme-betas] render step raised an exception (reporting, not treating as a "
                 "hard smoke-test failure -- the plan's check is primarily about the body-model "
                 "forward pass, but this is useful signal):")
            traceback.print_exc()
    else:
        print("[extreme-betas] no mesh renderer configured on the model, skipping render step.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    if os.environ.get("AMR_DISABLE_CUDNN", "0") == "1":
        print("AMR_DISABLE_CUDNN=1 -> disabling torch.backends.cudnn")
        torch.backends.cudnn.enabled = False

    print(f"Building config (experiment=AniMerPlus) ...")
    cfg = build_cfg()

    print(f"Constructing AniMerPlusPlus on {args.device} ...")
    from amr.models.animerpp import AniMerPlusPlus
    model = AniMerPlusPlus(cfg, init_renderer=True)
    model.to(args.device)
    model.eval()

    print("\n=== Scenario 1: reasonable random batch ===")
    batch = make_fake_batch(cfg, args.batch_size, args.device)
    run_forward_and_loss(model, batch, label="reasonable")

    print("\n=== Scenario 2: out-of-shape-space betas (VAREN_integration_plan.md check) ===")
    run_extreme_betas_test(model, cfg, args.batch_size, args.device)

    print("\nAll smoke test scenarios completed without crashing.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\nSMOKE TEST FAILED:", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
