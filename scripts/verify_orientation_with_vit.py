#!/usr/bin/env python
"""
Orientation sanity check WITH the real ViT-MoE backbone + VAREN head (the
full learnable pipeline), without waiting for a full training run.

Overfits the real AniMerPlusPlus model -- real backbone, real head, real
VAREN body model, real gradients -- on a small handful of real training
samples for a few hundred steps (a standard "can it overfit one batch"
debugging technique). This is the complement to verify_orientation_no_vit.py:
that script checks the body-model/axis plumbing in isolation (no network
involved); this script checks that the *learnable* path -- backbone
gradients through to the rendered keypoints -- actually converges toward the
correct orientation.

  PASS here  = loss drops sharply and the predicted 2D keypoints end up close
               to the real ones. The learnable pipeline is wired correctly.
  FAIL here  = loss plateaus high, or predictions don't converge to match the
               image, even on a tiny handful of examples it should be able to
               memorize. That's a strong, fast (minutes, not ~2000 steps)
               signal something in what the network is being asked to learn
               is broken (e.g. the rotation representation -- see
               .agent/Checks.md) -- separate from the plumbing
               verify_orientation_no_vit.py checks.

Backbone pretrained weights are skipped by default (random init) -- this
check is about whether gradients flow correctly and the geometry is
learnable at all, not about final training quality, so pretrained features
aren't needed and loading them (a multi-GB file) would only slow this down.

Usage:
    python scripts/verify_orientation_with_vit.py \
        --json-file /path/to/horse_dataset/train.json \
        --root-image /path/to/root_image_dir \
        [--num-samples 2] [--steps 300] [--device cuda] \
        [--varen-model-path /path/to/VAREN/models]

Exits non-zero if the loss doesn't drop enough or the final 2D alignment
error is too large.
"""
import argparse
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
from torch.utils.data import DataLoader


def build_cfg(json_file, root_image, varen_model_path, extra_overrides=None):
    from hydra import compose, initialize_config_dir

    extra_overrides = extra_overrides or []
    config_dir = str(root / "amr" / "configs_hydra")
    overrides = [
        "experiment=AniMerPlus",
        "trainer=gpu",
        "launcher=local",
        # Skip the multi-GB pretrained backbone checkpoint -- this check is about
        # whether gradients/geometry are wired correctly, not final quality, and
        # random-init trains just as well for a tiny-batch overfit sanity check.
        "MODEL.BACKBONE.PRETRAINED_WEIGHTS=null",
    ] + extra_overrides
    with initialize_config_dir(version_base="1.2", config_dir=config_dir):
        cfg = compose(config_name="train.yaml", overrides=overrides)
    if varen_model_path:
        cfg.VAREN.MODEL_PATH = varen_model_path  # modifying an existing key -- no defrost needed (that's yacs, not OmegaConf)
    return cfg


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--json-file", required=True, help="Path to a horse_dataset-format JSON (e.g. train.json)")
    p.add_argument("--root-image", required=True, help="ROOT_IMAGE the JSON's img_path/mask_path are relative to")
    p.add_argument("--varen-model-path", default=None,
                  help="Override cfg.VAREN.MODEL_PATH (the directory holding VAREN.pkl) -- "
                       "the config's default is a cluster path, so this almost always needs "
                       "setting explicitly unless you're already running where that path resolves.")
    p.add_argument("--num-samples", type=int, default=2,
                  help="How many real samples to overfit on (kept small on purpose -- this is "
                       "a 'can it memorize a handful of examples' check, not real training)")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--loss-drop-fraction", type=float, default=0.5,
                  help="Require final loss < this fraction of the first-step loss to PASS")
    p.add_argument("--pixel-error-threshold-frac", type=float, default=0.1,
                  help="Require final mean 2D keypoint error < this fraction of IMAGE_SIZE to PASS")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = build_cfg(args.json_file, args.root_image, args.varen_model_path)

    from amr.datasets.varen_dataset import VARENEvaluationDataset
    from amr.models.animerpp import AniMerPlusPlus
    from amr.utils import recursive_to

    dataset = VARENEvaluationDataset(
        root_image=args.root_image,
        json_file=args.json_file,
        augm_config=cfg.DATASETS.CONFIG,  # unused in practice: this dataset hardcodes is_train=False
        focal_length=cfg.VAREN.get("FOCAL_LENGTH", 1000),
        image_size=cfg.MODEL.IMAGE_SIZE,
        num_joints=cfg.VAREN.get("NUM_JOINTS", 37),
        num_betas=cfg.VAREN.get("NUM_BETAS", 39),
    )
    dataset.data['data'] = dataset.data['data'][:args.num_samples]
    n = len(dataset)
    if n == 0:
        print(f"No samples found in {args.json_file}", file=sys.stderr)
        sys.exit(2)
    print(f"Overfitting on {n} real sample(s) from {args.json_file} for {args.steps} steps on {args.device} ...")

    loader = DataLoader(dataset, batch_size=n, shuffle=False)
    batch = recursive_to(next(iter(loader)), args.device)

    model = AniMerPlusPlus(cfg, init_renderer=False)
    model.to(args.device)
    model.train()

    optimizer = torch.optim.Adam(model.get_parameters(), lr=args.lr)

    losses = []
    print_every = max(1, args.steps // 10)
    for step in range(args.steps):
        optimizer.zero_grad()
        output = model.forward_step(batch, train=True)
        loss = model.compute_loss(batch, output, train=True)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if step % print_every == 0 or step == args.steps - 1:
            print(f"  step {step:4d}: loss = {loss.item():.5f}")

    first_loss, last_loss = losses[0], float(np.mean(losses[-print_every:]))
    loss_target = first_loss * args.loss_drop_fraction
    loss_ok = last_loss < loss_target
    print(f"\nloss: {first_loss:.5f} -> {last_loss:.5f} "
         f"(needed < {loss_target:.5f} to pass) [{'PASS' if loss_ok else 'FAIL'}]")

    model.eval()
    with torch.no_grad():
        output = model.forward_step(batch, train=False)
    pred_2d = output['varen_output']['pred_keypoints_2d'].cpu().numpy()
    gt_2d = batch['keypoints_2d'][..., :2].cpu().numpy()
    # both are in the [-0.5, 0.5]-normalized-by-patch-width units get_example() uses --
    # scale back to pixels (of the IMAGE_SIZE-square input patch) to report something readable
    px_err = float(np.linalg.norm(pred_2d - gt_2d, axis=-1).mean() * cfg.MODEL.IMAGE_SIZE)
    px_threshold = args.pixel_error_threshold_frac * cfg.MODEL.IMAGE_SIZE
    align_ok = px_err < px_threshold
    print(f"final mean 2D keypoint error: {px_err:.1f}px on a {cfg.MODEL.IMAGE_SIZE}px patch "
         f"(needed < {px_threshold:.1f}px to pass) [{'PASS' if align_ok else 'FAIL'}]")

    ok = loss_ok and align_ok
    print()
    if ok:
        print("PASS -- the full learnable pipeline (backbone -> head -> VAREN -> projection) "
             "can converge toward the correct orientation on real data.")
    else:
        print("FAIL -- the pipeline did not converge the way a correctly-wired one should, even "
             "on a handful of memorizable examples. If verify_orientation_no_vit.py PASSED but "
             "this FAILS, the issue is specifically in what the network is being asked to learn "
             "(e.g. JOINT_REP / the rotation representation), not the body-model plumbing.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\nVERIFY-ORIENTATION (with ViT) SCRIPT FAILED WITH AN EXCEPTION:", file=sys.stderr)
        traceback.print_exc()
        sys.exit(3)
