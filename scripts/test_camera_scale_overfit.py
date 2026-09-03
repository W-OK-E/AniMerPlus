#!/usr/bin/env python
"""
Overfit-a-tiny-batch check for the camera-scale fix (2026-08-25 debugging session).

Same "can it memorize a handful of real samples" technique as
verify_orientation_with_vit.py, but:
  - matches real training's backbone config (run.sh's
    MODEL.BACKBONE.FREEZE_ATTN/FREEZE_FFN/FROZEN_STAGES/USE_CLS + the real
    pretrained checkpoint), which verify_orientation_with_vit.py's own
    build_cfg() does not apply. Defaults now match run.sh's full-unfreeze +
    discriminative-LR setup (FREEZE_ATTN/FREEZE_FFN=false, FROZEN_STAGES=-1 --
    nothing frozen; TRAIN.BACKBONE_LR_GROUPS in AniMerPlus.yaml governs the
    effective per-block LR instead -- blocks <=10 at 0.01x, <=25 at 0.1x, 26+
    and the heads at the full LR); override via --freeze-attn/--freeze-ffn/
    --frozen-stages, e.g. pass --freeze-attn true --freeze-ffn true to go back
    to the fully-frozen backbone if it OOMs a 12GB card.
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
  - saves a checkpoint (model+optimizer state) every --checkpoint-every steps
    (default: same as --log-every) to --checkpoint-dir, each paired with a
    render snapshot on the training batch at that step (step_NNNNNN.pt +
    step_NNNNNN_render.png) -- so you can watch alignment evolve over the run,
    not just see the final result. Pass --checkpoint-every 0 to disable.
  - can resume from one of those checkpoints via --resume-from
    checkpoint_dir/step_NNNNNN.pt -- --steps is the TARGET total, so resuming
    a step-300 checkpoint with --steps 800 runs 500 more steps, not 800 more.
    Must use the same sample/model config it was saved with.
  - for each HOLDOUT sample (the final val-set-style check), also saves pred +
    GT VAREN params as JSON in the exact same flat schema as
    VAREN/examples/example_params.json (global_orient/pose/betas, axis-angle)
    to --params-out-dir, so they can be loaded and compared independently in
    trimesh/blender -- and (if --render) a semi-transparent wireframe overlay
    PNG per sample, which shows alignment more precisely than the solid-shaded
    mesh in the main grid render.

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
        [--checkpoint-dir camera_scale_overfit_checkpoints] [--checkpoint-every 50]
        [--resume-from PATH] [--freeze-attn false] [--freeze-ffn false] [--frozen-stages -1]
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
    p.add_argument("--freeze-attn", type=lambda s: s.lower() == 'true', default=False,
                  help="MODEL.BACKBONE.FREEZE_ATTN (matches run.sh's default: false)")
    p.add_argument("--freeze-ffn", type=lambda s: s.lower() == 'true', default=False,
                  help="MODEL.BACKBONE.FREEZE_FFN (matches run.sh's default: false)")
    p.add_argument("--frozen-stages", type=int, default=-1,
                  help="MODEL.BACKBONE.FROZEN_STAGES (matches run.sh's default: -1, i.e. nothing "
                       "frozen -- TRAIN.BACKBONE_LR_GROUPS in AniMerPlus.yaml governs the "
                       "effective per-block LR instead). Pass e.g. 27 to freeze blocks 1..27, "
                       "leaving block 0 and blocks after N trainable. Only takes effect when "
                       "--freeze-attn/--freeze-ffn are both false, since those override every "
                       "block unconditionally when true.")
    p.add_argument("--num-samples", type=int, default=10)
    p.add_argument("--num-holdout-samples", type=int, default=10,
                  help="Samples NOT used for training, for the final results/render "
                       "(checks generalization, not just memorization)")
    p.add_argument("--seed", type=int, default=None, help="torch.manual_seed for reproducibility")
    p.add_argument("--sample-offset", type=float, default=0.0,
                  help="Fraction-of-bucket offset for spread sampling (0.0 or 0.5 were used "
                       "in the original debugging session to get two different sample sets)")
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--checkpoint-dir", default="camera_scale_overfit_checkpoints",
                  help="Folder for periodic model checkpoints + their render snapshots "
                       "(on the training batch, to watch alignment evolve over the run)")
    p.add_argument("--checkpoint-every", type=int, default=None,
                  help="Steps between checkpoints (default: same as --log-every). "
                       "Pass 0 to disable checkpointing.")
    p.add_argument("--resume-from", default=None,
                  help="Path to a step_NNNNNN.pt checkpoint to resume training from. "
                       "--steps is the TARGET total step count, not additional steps -- "
                       "e.g. resuming a step-300 checkpoint with --steps 800 runs 500 more "
                       "steps. Must be resumed with the same model/sample config it was "
                       "saved with (same --num-samples etc.), otherwise shapes won't match.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--disable-fix", action="store_true",
                  help="Monkeypatch the OLD unconstrained camera-scale formula back in, "
                       "for a side-by-side comparison against the real (fixed) code.")
    p.add_argument("--pass-threshold-px", type=float, default=25.6)
    p.add_argument("--render", dest="render", action="store_true", default=True)
    p.add_argument("--no-render", dest="render", action="store_false")
    p.add_argument("--render-out", default="camera_scale_overfit_render.png")
    p.add_argument("--params-out-dir", default="camera_scale_overfit_params",
                  help="Per-holdout-sample pred/GT VAREN params (VAREN/examples/"
                       "example_params.json schema) + wireframe overlay PNGs")
    return p.parse_args()


def build_cfg(json_file, root_image, varen_model_path, pretrained_weights,
              freeze_attn=False, freeze_ffn=False, frozen_stages=-1):
    from hydra import compose, initialize_config_dir

    overrides = [
        "experiment=AniMerPlus",
        "trainer=gpu",
        "launcher=local",
        f"MODEL.BACKBONE.FREEZE_ATTN={str(freeze_attn).lower()}",
        f"MODEL.BACKBONE.FREEZE_FFN={str(freeze_ffn).lower()}",
        f"MODEL.BACKBONE.FROZEN_STAGES={frozen_stages}",
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
    return all_data, available, idxs, yaws


def denormalize_images(images):
    """batch['img'] is ImageNet-normalized; undo that for rendering, matching
    AniMerPlusPlus.tensorboard_logging's own normalization exactly."""
    std = torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(1, 3, 1, 1)
    mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1, 3, 1, 1)
    return images * std + mean


def render_wireframe_overlay(vertices, camera_translation, image, focal_length, faces,
                             alpha=0.45, color=(0.15, 0.9, 0.35)):
    """Semi-transparent wireframe of the predicted mesh over the real image --
    edges show alignment more precisely than a solid shaded mesh, and staying
    transparent keeps the underlying photo visible for comparison."""
    import pyrender
    import trimesh
    from amr.utils.mesh_renderer import create_raymond_lights

    renderer = pyrender.OffscreenRenderer(viewport_width=image.shape[1], viewport_height=image.shape[0])
    material = pyrender.MetallicRoughnessMaterial(metallicFactor=0.0, alphaMode='OPAQUE',
                                                  baseColorFactor=(*color, 1.0), wireframe=True)
    camera_translation = camera_translation.copy()
    camera_translation[0] *= -1.
    mesh = trimesh.Trimesh(vertices.copy(), faces.copy())
    rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
    mesh.apply_transform(rot)
    mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(1.0, 1.0, 1.0))
    scene.add(mesh, 'mesh')
    camera_pose = np.eye(4)
    camera_pose[:3, 3] = camera_translation
    camera_center = [image.shape[1] / 2., image.shape[0] / 2.]
    camera = pyrender.IntrinsicsCamera(fx=focal_length, fy=focal_length,
                                       cx=camera_center[0], cy=camera_center[1], zfar=1000)
    scene.add(camera, pose=camera_pose)
    for node in create_raymond_lights():
        scene.add_node(node)

    color_buf, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA | pyrender.RenderFlags.ALL_WIREFRAME)
    renderer.delete()
    color_buf = color_buf.astype(np.float32) / 255.0
    wireframe_mask = (color_buf[:, :, -1] > 0)[:, :, np.newaxis] * alpha
    output_img = color_buf[:, :, :3] * wireframe_mask + image * (1 - wireframe_mask)
    return output_img.astype(np.float32)


def varen_params_to_json_dict(global_orient, body_pose, betas, is_axis_angle):
    """Matches VAREN/examples/example_params.json's flat schema exactly:
    {global_orient: [3], pose: [NUM_JOINTS*3], betas: [NUM_BETAS]}, all axis-angle."""
    if not is_axis_angle:
        from pytorch3d.transforms import matrix_to_axis_angle
        global_orient = matrix_to_axis_angle(global_orient.reshape(-1, 3, 3)).reshape(-1)
        body_pose = matrix_to_axis_angle(body_pose.reshape(-1, 3, 3)).reshape(-1)
    else:
        global_orient = global_orient.reshape(-1)
        body_pose = body_pose.reshape(-1)
    return {
        'global_orient': global_orient.detach().cpu().tolist(),
        'pose': body_pose.detach().cpu().tolist(),
        'betas': betas.reshape(-1).detach().cpu().tolist(),
    }


def save_sample_outputs(output, batch, sample_idx, out_dir, prefix, is_axis_angle_pred, faces,
                        image_np, focal_length, render):
    """For one sample: pred + GT params as VAREN-example-format JSON, and (if
    render) a wireframe overlay PNG -- everything needed to inspect this one
    prediction independently in trimesh/blender, without the rest of the grid."""
    os.makedirs(out_dir, exist_ok=True)
    pred_params = output['varen_output']['pred_params']
    pred_dict = varen_params_to_json_dict(
        pred_params['global_orient'][sample_idx], pred_params['body_pose'][sample_idx],
        pred_params['betas'][sample_idx], is_axis_angle_pred)
    with open(os.path.join(out_dir, f"{prefix}_pred_params.json"), 'w') as f:
        json.dump(pred_dict, f)

    gt_params = batch['varen_params']
    gt_dict = varen_params_to_json_dict(
        gt_params['global_orient'][sample_idx], gt_params['body_pose'][sample_idx],
        gt_params['betas'][sample_idx], is_axis_angle=True)  # dataset GT is always axis-angle
    with open(os.path.join(out_dir, f"{prefix}_gt_params.json"), 'w') as f:
        json.dump(gt_dict, f)

    if render:
        vertices = output['varen_output']['pred_vertices'][sample_idx].detach().float().cpu().numpy()
        cam_t = output['varen_output']['pred_cam_t'][sample_idx].detach().float().cpu().numpy()
        wireframe = render_wireframe_overlay(vertices, cam_t, image_np, focal_length, faces)
        from PIL import Image
        Image.fromarray((wireframe.clip(0, 1) * 255).astype(np.uint8)).save(
            os.path.join(out_dir, f"{prefix}_wireframe.png"))


def save_checkpoint(model, optimizer, step, checkpoint_dir, batch, cfg, render, scheduler=None):
    """Saves model+optimizer(+scheduler) state, and (if render) a render
    snapshot of the current predictions on the training batch, both named by step."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, f"step_{step:06d}.pt")
    torch.save({'step': step, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
               'scheduler': scheduler.state_dict() if scheduler is not None else None}, ckpt_path)

    if not render:
        print(f"  [checkpoint] saved {ckpt_path}")
        return

    model.eval()
    with torch.no_grad():
        output = model.forward_step(batch, train=False)
        model.compute_loss(batch, output, train=False)
        rend_imgs = model.tensorboard_logging(batch, output, step_count=step,
                                              train=False, write_to_summary_writer=False)
    model.train()
    render_path = os.path.join(checkpoint_dir, f"step_{step:06d}_render.png")
    try:
        from torchvision.utils import save_image
        save_image(rend_imgs, render_path)
        print(f"  [checkpoint] saved {ckpt_path} + {render_path}")
    except Exception:
        import traceback
        print(f"  [checkpoint] saved {ckpt_path}, but render failed (not fatal):", file=sys.stderr)
        traceback.print_exc()


def pick_holdout_samples(all_data, available, train_idxs, num_samples):
    """Pick samples from `available` that were NOT used for training, spread
    across the remaining pool the same way pick_available_samples does."""
    pool = [i for i in available if i not in set(train_idxs)]
    if not pool:
        return [], []
    n = min(num_samples, len(pool))
    idxs = [pool[int(k * len(pool) / n)] for k in range(n)]
    yaws = [float(np.degrees(all_data[i]['pose'][2])) for i in idxs]
    return idxs, yaws


def main():
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)
    torch.backends.cudnn.enabled = False

    cfg = build_cfg(args.json_file, args.root_image, args.varen_model_path, args.pretrained_weights,
                    freeze_attn=args.freeze_attn, freeze_ffn=args.freeze_ffn, frozen_stages=args.frozen_stages)
    cfg.TRAIN.LR = args.lr  # so --lr still works now that the optimizer comes from configure_optimizers()

    from amr.datasets.varen_dataset import VARENEvaluationDataset
    from amr.models.animerpp import AniMerPlusPlus
    from amr.utils import recursive_to

    all_data, available, idxs, yaws = pick_available_samples(args.json_file, args.root_image, args.num_samples, args.sample_offset)
    # print("train sample indices:", idxs)
    # print("train yaw angles (deg):", [round(y, 1) for y in yaws])

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
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    batch = recursive_to(next(iter(loader)), args.device)

    model = AniMerPlusPlus(cfg, init_renderer=args.render)
    model.to(args.device)
    model.train()

    if args.disable_fix:
        print("*** --disable-fix: using the OLD unconstrained camera-scale formula ***")
        model.forward_one_parametric_model = types.MethodType(old_unconstrained_forward_one_parametric_model, model)

    # Use the model's own configure_optimizers() -- NOT a bespoke plain-Adam
    # loop -- so this matches real training's optimizer exactly: AdamW +
    # TRAIN.WEIGHT_DECAY, TRAIN.BACKBONE_LR_GROUPS discriminative per-block LR
    # (if set), and the warmup->cosine LR schedule. A flat single-LR Adam here
    # silently diverged from what run.sh actually trains with.
    optimizers, schedulers = model.configure_optimizers()
    optimizer = optimizers[0]
    # DEBUG: print optimizer param groups
    print(f"[DEBUG] optimizer param_groups: {len(optimizer.param_groups)} groups")
    for i, g in enumerate(optimizer.param_groups):
        print(f"  group {i}: lr={g['lr']:.2e}, weight_decay={g.get('weight_decay', 0)}, params={len(g['params'])}")
    scheduler = schedulers[0] if schedulers else None
    checkpoint_every = args.log_every if args.checkpoint_every is None else args.checkpoint_every

    start_step = 0
    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location=args.device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        if scheduler is not None and ckpt.get('scheduler') is not None:
            scheduler.load_state_dict(ckpt['scheduler'])
        start_step = ckpt['step'] + 1
        print(f"resumed from {args.resume_from} at step {ckpt['step']} -> continuing from step {start_step}")
        if start_step >= args.steps:
            print(f"start_step ({start_step}) >= --steps ({args.steps}), nothing left to train -- "
                 "skipping straight to final eval/render.")

    for step in range(start_step, args.steps):
        optimizer.zero_grad()
        output = model.forward_step(batch, train=True)
        loss = model.compute_loss(batch, output, train=True)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if step % args.log_every == 0 or step == args.steps - 1:
            comps = {k: round(v.item(), 3) for k, v in output['losses'].items()}
            cam_z = output['varen_output']['pred_cam_t'][:, 2].detach().cpu().numpy()
            neg = int((cam_z < 0).sum())
            print(f"step {step:4d}: loss={comps['loss']:.3f} kp2d={comps.get('loss_varen_keypoints_2d', 0):.2f} "
                 f"cam_z_min={cam_z.min():.1f} cam_z_max={cam_z.max():.1f} cam_z_neg_count={neg}")
        if checkpoint_every > 0 and (step % checkpoint_every == 0 or step == args.steps - 1):
            save_checkpoint(model, optimizer, step, args.checkpoint_dir, batch, cfg, args.render, scheduler=scheduler)

    # Final results + render run on held-out samples (never seen during the
    # overfit loop above) -- checks generalization, not just memorization.
    holdout_idxs, holdout_yaws = pick_holdout_samples(all_data, available, idxs, args.num_holdout_samples)
    if not holdout_idxs:
        print("\nNo held-out samples available (dataset too small) -- skipping final eval/render.")
        return
    print("\nholdout sample indices:", holdout_idxs)
    print("holdout yaw angles (deg):", [round(y, 1) for y in holdout_yaws])

    holdout_dataset = VARENEvaluationDataset(
        root_image=args.root_image,
        json_file=args.json_file,
        augm_config=cfg.DATASETS.CONFIG,
        focal_length=cfg.VAREN.get("FOCAL_LENGTH", 1000),
        image_size=cfg.MODEL.IMAGE_SIZE,
        num_joints=cfg.VAREN.get("NUM_JOINTS", 37),
        num_betas=cfg.VAREN.get("NUM_BETAS", 39),
    )
    holdout_dataset.data['data'] = [all_data[i] for i in holdout_idxs]
    n_holdout = len(holdout_idxs)
    holdout_loader = DataLoader(holdout_dataset, batch_size=32, shuffle=False)
    holdout_batch = recursive_to(next(iter(holdout_loader)), args.device)

    model.eval()
    with torch.no_grad():
        output = model.forward_step(holdout_batch, train=False)
    pred_2d = output['varen_output']['pred_keypoints_2d']
    gt_2d = holdout_batch['keypoints_2d'][..., :2]
    per_sample_px = ((pred_2d - gt_2d).norm(dim=-1).mean(dim=-1) * cfg.MODEL.IMAGE_SIZE).detach().cpu().numpy()
    final_cam_z = output['varen_output']['pred_cam_t'][:, 2].detach().cpu().numpy()

    # Scale check: mean pelvis-relative keypoint distance, pred vs GT (same
    # quantity the new SCALE loss supervises). Ratio < 1 means predictions are
    # smaller than ground truth.
    pred_3d = output['varen_output']['pred_keypoints_3d']
    gt_3d = holdout_batch['keypoints_3d']
    gt_conf = gt_3d[:, :, -1]
    pred_rel = pred_3d - pred_3d[:, 0:1, :]
    gt_rel = gt_3d[:, :, :-1] - gt_3d[:, 0:1, :-1]
    pred_scale = ((pred_rel.norm(dim=-1) * gt_conf).sum(dim=1) / gt_conf.sum(dim=1).clamp(min=1)).detach().cpu().numpy()
    gt_scale = ((gt_rel.norm(dim=-1) * gt_conf).sum(dim=1) / gt_conf.sum(dim=1).clamp(min=1)).detach().cpu().numpy()
    scale_ratio = pred_scale / gt_scale

    print()
    print(f"=== per-sample HOLDOUT results ({'OLD formula' if args.disable_fix else 'fixed formula'}) ===")
    for i in range(n_holdout):
        print(f"sample {i}: yaw={holdout_yaws[i]:7.1f} deg  final_2d_err={per_sample_px[i]:7.1f}px  "
             f"final_cam_z={final_cam_z[i]:8.1f}  scale_ratio(pred/gt)={scale_ratio[i]:.3f}")
    n_pass = int((per_sample_px < args.pass_threshold_px).sum())
    print(f"{n_pass}/{n_holdout} holdout samples under {args.pass_threshold_px}px threshold")
    print(f"mean scale_ratio: {scale_ratio.mean():.3f} (1.0 = correct size, <1.0 = predictions too small)")

    if args.render:
        try:
            model.compute_loss(holdout_batch, output, train=False)
            rend_imgs = model.tensorboard_logging(holdout_batch, output, step_count=args.steps,
                                                  train=False, write_to_summary_writer=False)
            from torchvision.utils import save_image
            save_image(rend_imgs, args.render_out)
            print(f"\nSaved HOLDOUT render (per sample: image | mesh front | mesh side | pred keypoints "
                 f"| GT keypoints) to {args.render_out}")
        except Exception:
            import traceback
            print("\n[render skipped] rendering raised an exception -- not a hard failure, "
                 "the numeric results above are unaffected. Traceback:", file=sys.stderr)
            traceback.print_exc()

    # Per-sample outputs for independent inspection outside this script: pred +
    # GT params in the same flat schema as VAREN/examples/example_params.json
    # (load directly in trimesh/blender), plus a semi-transparent wireframe
    # overlay per sample (edges read alignment more precisely than solid shading).
    try:
        images_np = denormalize_images(holdout_batch['img']).permute(0, 2, 3, 1).detach().cpu().numpy()
        focal_length = float(cfg.VAREN.get("FOCAL_LENGTH", 1000))
        is_axis_angle_pred = model.varen_head.joint_rep_type == 'aa'
        faces = model.varen.faces
        for i in range(n_holdout):
            prefix = f"holdout_{i:03d}"
            save_sample_outputs(output, holdout_batch, i, args.params_out_dir, prefix,
                               is_axis_angle_pred, faces, images_np[i], focal_length, args.render)
        print(f"\nSaved per-sample pred/GT params (+ wireframe overlays, if --render) "
             f"for {n_holdout} holdout samples to {args.params_out_dir}/")
    except Exception:
        import traceback
        print("\n[per-sample outputs skipped] raised an exception -- not a hard failure, "
             "the results above are unaffected. Traceback:", file=sys.stderr)
        traceback.print_exc()


if __name__ == "__main__":
    main()
