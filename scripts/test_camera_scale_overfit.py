#!/usr/bin/env python
"""
Overfit-a-tiny-batch check for the camera-scale fix (2026-08-25 debugging session).

Same "can it memorize a handful of real samples" technique as
verify_orientation_with_vit.py, but:
  - matches real training's backbone-freeze config (run.sh's
    MODEL.BACKBONE.FREEZE_ATTN/FREEZE_FFN/USE_CLS + the real pretrained
    checkpoint), which verify_orientation_with_vit.py's own build_cfg() does not
    apply -- without it the full ViT-MoE backbone is trainable and OOMs on a
    12GB card.
  - spreads sample selection across the whole dataset (not just the first N),
    and skips samples whose image/mask files aren't present locally.
  - disables cuDNN (this machine's GPU/cuDNN combo raises "GET was unable to
    find an engine" on plain nn.Conv2d otherwise -- same issue
    smoke_test_varen.py works around via AMR_DISABLE_CUDNN).
  - prints the per-component loss breakdown and pred_cam_t's z-component
    (camera depth) every --log-every steps, and a per-sample final summary
    (2D pixel error + final camera depth) at the end.
  - renders by default (per sample: input image | mesh front | mesh side |
    pred 2D keypoints | GT 2D keypoints), reusing the exact same
    AniMerPlusPlus.tensorboard_logging / MeshRenderer.visualize_tensorboard
    call the real training loop logs to TensorBoard with.

The fix itself (amr/models/animerpp.py, forward_one_parametric_model): the
predicted camera scale used to be used raw as a divisor
(2*focal/(IMAGE_SIZE*pred_cam[:,0]+1e-9)), with no positivity constraint --
it could cross zero, flipping the camera to a negative depth, or blow up
unboundedly. Now it's passed through softplus first so it can't reach zero or
go negative. Pass --disable-fix to temporarily monkeypatch the OLD unconstrained
formula back in for a side-by-side comparison (does not touch the file on disk).

Usage:
    python scripts/test_camera_scale_overfit.py [--num-samples 10] [--steps 800]
        [--lr 1e-4] [--log-every 50] [--device cuda] [--disable-fix]
        [--render-out camera_scale_overfit_render.png] [--no-render]
        [--json-file ...] [--root-image ...] [--varen-model-path ...]

Needs the animer2 micromamba env active (or run via
`micromamba run -n animer2 python scripts/test_camera_scale_overfit.py ...`).
"""
import argparse
import json
import os
import sys
import types

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


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--json-file", default="/home/om/mpi/animer_train_data/horse_dataset_textured/train.json")
    p.add_argument("--root-image", default="/home/om/mpi/animer_train_data/batches")
    p.add_argument("--varen-model-path", default="/home/om/mpi/VAREN/models")
    p.add_argument("--pretrained-weights", default="data/AniMerPlus/checkpoint.ckpt",
                  help="Real pretrained AniMer+ checkpoint (matches run.sh). Pass empty string "
                       "('') for a fast random-init smoke check instead.")
    p.add_argument("--num-samples", type=int, default=10)
    p.add_argument("--seed", type=int, default=None, help="torch.manual_seed for reproducibility")
    p.add_argument("--sample-offset", type=float, default=0.0,
                  help="Fraction-of-bucket offset for spread sampling (0.0 or 0.5 were used "
                       "in the original debugging session to get two different sample sets)")
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--disable-fix", action="store_true",
                  help="Monkeypatch the OLD unconstrained camera-scale formula back in, "
                       "for a side-by-side comparison against the real (fixed) code.")
    p.add_argument("--pass-threshold-px", type=float, default=25.6)
    p.add_argument("--render", dest="render", action="store_true", default=True)
    p.add_argument("--no-render", dest="render", action="store_false")
    p.add_argument("--render-out", default="camera_scale_overfit_render.png")
    return p.parse_args()


def build_cfg(json_file, root_image, varen_model_path, pretrained_weights):
    from hydra import compose, initialize_config_dir

    overrides = [
        "experiment=AniMerPlus",
        "trainer=gpu",
        "launcher=local",
        "MODEL.BACKBONE.FREEZE_ATTN=true",
        "MODEL.BACKBONE.FREEZE_FFN=true",
        "MODEL.BACKBONE.USE_CLS=false",
    ]
    if pretrained_weights:
        overrides.append(f"MODEL.BACKBONE.PRETRAINED_WEIGHTS={pretrained_weights}")
    else:
        overrides.append("MODEL.BACKBONE.PRETRAINED_WEIGHTS=null")
    config_dir = str(root / "amr" / "configs_hydra")
    with initialize_config_dir(version_base="1.2", config_dir=config_dir):
        cfg = compose(config_name="train.yaml", overrides=overrides)
    cfg.VAREN.MODEL_PATH = varen_model_path
    cfg.DATASETS.HORSE.ROOT_IMAGE = root_image
    cfg.DATASETS.HORSE.JSON_FILE.TRAIN = json_file
    return cfg


def old_unconstrained_forward_one_parametric_model(self, focal_length, features, head, parametric_model):
    """Byte-identical to the pre-fix forward_one_parametric_model, for --disable-fix comparisons."""
    from amr.models.animerpp import _varen_native_to_camera_frame
    from amr.utils.geometry import perspective_projection

    batch_size = features.shape[0]
    pred_params, pred_cam, _ = head(features)
    output = {}
    output['pred_cam'] = pred_cam
    output['pred_params'] = {k: v.clone() for k, v in pred_params.items()}

    pred_cam_t = torch.stack([pred_cam[:, 1],
                              pred_cam[:, 2],
                              2 * focal_length[:, 0] / (self.cfg.MODEL.IMAGE_SIZE * pred_cam[:, 0] + 1e-9)], dim=-1)
    output['pred_cam_t'] = pred_cam_t
    output['focal_length'] = focal_length

    pred_params['betas'] = pred_params['betas'].reshape(batch_size, -1)
    betas = pred_params['betas']
    is_axis_angle = head.joint_rep_type == 'aa'
    if is_axis_angle:
        global_orient = pred_params['global_orient'].reshape(batch_size, 3)
        body_pose = pred_params['body_pose'].reshape(batch_size, -1)
    else:
        global_orient = pred_params['global_orient'].reshape(batch_size, 1, 3, 3)
        body_pose = pred_params['body_pose'].reshape(batch_size, -1, 3, 3)
    parametric_model_output = parametric_model(global_orient=global_orient, body_pose=body_pose,
                                               betas=betas, transl=None, pose2rot=is_axis_angle)
    parametric_model_output.vertices = _varen_native_to_camera_frame(parametric_model_output.vertices)
    if getattr(parametric_model_output, 'surface_keypoints', None) is not None:
        parametric_model_output.surface_keypoints = _varen_native_to_camera_frame(parametric_model_output.surface_keypoints)
    if getattr(parametric_model_output, 'joints', None) is not None:
        parametric_model_output.joints = _varen_native_to_camera_frame(parametric_model_output.joints)

    surface_keypoints = getattr(parametric_model_output, 'surface_keypoints', None)
    pred_keypoints_3d = surface_keypoints if surface_keypoints is not None else parametric_model_output.joints
    pred_vertices = parametric_model_output.vertices
    output['pred_keypoints_3d'] = pred_keypoints_3d.reshape(batch_size, -1, 3)
    output['pred_vertices'] = pred_vertices.reshape(batch_size, -1, 3)
    pred_cam_t = pred_cam_t.reshape(-1, 3)
    focal_length = focal_length.reshape(-1, 2)
    pred_keypoints_2d = perspective_projection(pred_keypoints_3d, translation=pred_cam_t,
                                               focal_length=focal_length / self.cfg.MODEL.IMAGE_SIZE)
    output['pred_keypoints_2d'] = pred_keypoints_2d.reshape(batch_size, -1, 2)
    return output


def pick_available_samples(json_file, root_image, num_samples, offset):
    with open(json_file) as f:
        all_data = json.load(f)['data']
    n_total = len(all_data)
    available = [i for i in range(n_total)
                if os.path.exists(os.path.join(root_image, all_data[i]['img_path']))
                and os.path.exists(os.path.join(root_image, all_data[i]['mask_path']))]
    print(f"{len(available)}/{n_total} samples have both image+mask present locally")
    if len(available) <= num_samples:
        idxs = available
    else:
        idxs = [available[int((k + offset) * len(available) / num_samples)] for k in range(num_samples)]
    yaws = [float(np.degrees(all_data[i]['pose'][2])) for i in idxs]
    return all_data, idxs, yaws


def main():
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)
    torch.backends.cudnn.enabled = False

    cfg = build_cfg(args.json_file, args.root_image, args.varen_model_path, args.pretrained_weights)

    from amr.datasets.varen_dataset import VARENEvaluationDataset
    from amr.models.animerpp import AniMerPlusPlus
    from amr.utils import recursive_to

    all_data, idxs, yaws = pick_available_samples(args.json_file, args.root_image, args.num_samples, args.sample_offset)
    print("sample indices:", idxs)
    print("yaw angles (deg):", [round(y, 1) for y in yaws])

    dataset = VARENEvaluationDataset(
        root_image=args.root_image,
        json_file=args.json_file,
        augm_config=cfg.DATASETS.CONFIG,
        focal_length=cfg.VAREN.get("FOCAL_LENGTH", 1000),
        image_size=cfg.MODEL.IMAGE_SIZE,
        num_joints=cfg.VAREN.get("NUM_JOINTS", 37),
        num_betas=cfg.VAREN.get("NUM_BETAS", 39),
    )
    dataset.data['data'] = [all_data[i] for i in idxs]
    n_samples = len(idxs)
    loader = DataLoader(dataset, batch_size=n_samples, shuffle=False)
    batch = recursive_to(next(iter(loader)), args.device)

    model = AniMerPlusPlus(cfg, init_renderer=args.render)
    model.to(args.device)
    model.train()

    if args.disable_fix:
        print("*** --disable-fix: using the OLD unconstrained camera-scale formula ***")
        model.forward_one_parametric_model = types.MethodType(old_unconstrained_forward_one_parametric_model, model)

    optimizer = torch.optim.Adam(model.get_parameters(), lr=args.lr)

    for step in range(args.steps):
        optimizer.zero_grad()
        output = model.forward_step(batch, train=True)
        loss = model.compute_loss(batch, output, train=True)
        loss.backward()
        optimizer.step()
        if step % args.log_every == 0 or step == args.steps - 1:
            comps = {k: round(v.item(), 3) for k, v in output['losses'].items()}
            cam_z = output['varen_output']['pred_cam_t'][:, 2].detach().cpu().numpy()
            neg = int((cam_z < 0).sum())
            print(f"step {step:4d}: loss={comps['loss']:.3f} kp2d={comps.get('loss_varen_keypoints_2d', 0):.2f} "
                 f"cam_z_min={cam_z.min():.1f} cam_z_max={cam_z.max():.1f} cam_z_neg_count={neg}")

    model.eval()
    with torch.no_grad():
        output = model.forward_step(batch, train=False)
    pred_2d = output['varen_output']['pred_keypoints_2d']
    gt_2d = batch['keypoints_2d'][..., :2]
    per_sample_px = ((pred_2d - gt_2d).norm(dim=-1).mean(dim=-1) * cfg.MODEL.IMAGE_SIZE).detach().cpu().numpy()
    final_cam_z = output['varen_output']['pred_cam_t'][:, 2].detach().cpu().numpy()
    print()
    print(f"=== per-sample final results ({'OLD formula' if args.disable_fix else 'fixed formula'}) ===")
    for i in range(n_samples):
        print(f"sample {i}: yaw={yaws[i]:7.1f} deg  final_2d_err={per_sample_px[i]:7.1f}px  final_cam_z={final_cam_z[i]:8.1f}")
    n_pass = int((per_sample_px < args.pass_threshold_px).sum())
    print(f"{n_pass}/{n_samples} samples under {args.pass_threshold_px}px threshold")

    if args.render:
        try:
            model.compute_loss(batch, output, train=False)
            rend_imgs = model.tensorboard_logging(batch, output, step_count=args.steps,
                                                  train=False, write_to_summary_writer=False)
            from torchvision.utils import save_image
            save_image(rend_imgs, args.render_out)
            print(f"\nSaved render (per sample: image | mesh front | mesh side | pred keypoints "
                 f"| GT keypoints) to {args.render_out}")
        except Exception:
            import traceback
            print("\n[render skipped] rendering raised an exception -- not a hard failure, "
                 "the numeric results above are unaffected. Traceback:", file=sys.stderr)
            traceback.print_exc()


if __name__ == "__main__":
    main()
