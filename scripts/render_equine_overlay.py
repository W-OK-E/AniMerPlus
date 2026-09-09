#!/usr/bin/env python
"""
Render 4DEquine's predicted mesh overlaid on its input crop.

Produces the same two-panel [input | mesh overlay] figure AniMer's demo.py
writes (see symp_viz_dat/animer_out1/img_path_0.png), so the two methods can be
put side by side on a slide, using this repo's own MeshRenderer -- the same code
path AniMerPlusPlus.tensorboard_logging uses (animerpp.py:375-386).

4DEquine turns out to share this pipeline's crop-camera convention exactly:
solving its own `cam_t_z = 2*f / (IMAGE_SIZE * scale)` for f gives 5000.0 on
every sample, matching cfg.VAREN.FOCAL_LENGTH, so its stored `refined_cam_t`
can be handed straight to the renderer with no conversion.

The mesh rendered is refined_results.pt's stored `vertices`, not a mesh rebuilt
from the refined params: the stored tensor is 4DEquine's actual visual output
and includes `refined_tail_scale`, which the VAREN params schema cannot express.
(The metrics go the other way -- rebuilt from params with the tail excluded --
because there the two must be a like-for-like parameter comparison.)

USAGE:
    python scripts/render_equine_overlay.py \
        --root /home/om/mpi/symp/test_data/symp_viz_dat \
        --samples 1-10 --out-dir <root>/equine_render
"""
import argparse
import os

import numpy as np
import pyrootutils
import torch

root = pyrootutils.setup_root(
    search_from=__file__, indicator=[".git", "pyproject.toml"], pythonpath=True, dotenv=True)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_samples(spec: str) -> list[int]:
    """'1-10' or '1,3,7' -> list of indices."""
    out = []
    for part in spec.split(','):
        if '-' in part:
            lo, hi = part.split('-')
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def denormalize(img: torch.Tensor) -> np.ndarray:
    """(1,3,H,W) ImageNet-normalized -> (H,W,3) float in [0,1]."""
    x = img[0].permute(1, 2, 0).float().cpu().numpy()
    return np.clip(x * IMAGENET_STD + IMAGENET_MEAN, 0, 1)


def implied_focal(cam_scale: float, cam_t_z: float, image_size: int) -> float:
    """Invert cam_t_z = 2*f / (image_size * scale). Reported so a convention
    mismatch shows up as a number rather than a silently wrong overlay."""
    return float(cam_t_z * image_size * cam_scale / 2.0)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="symp_viz_dat directory")
    p.add_argument("--samples", default="1-10", help="e.g. '1-10' or '1,3,7'")
    p.add_argument("--out-dir", default=None, help="default: <root>/equine_render")
    p.add_argument("--varen-model-path", default="/home/om/mpi/VAREN/models")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--focal-length", type=float, default=5000.0)
    p.add_argument("--raw", action="store_true",
                   help="render the PRE-refinement AniMer prediction instead of the "
                        "post-optimised one, to isolate what post-refinement does")
    p.add_argument("--side-view", action="store_true",
                   help="add a third panel with the mesh rotated 90 degrees")
    args = p.parse_args()

    out_dir = args.out_dir or os.path.join(args.root, "equine_render")
    os.makedirs(out_dir, exist_ok=True)

    from omegaconf import OmegaConf
    from PIL import Image

    from amr.models.varen_wrapper import VAREN
    from amr.utils.mesh_renderer import MeshRenderer

    varen = VAREN(model_path=args.varen_model_path, num_betas=39)
    cfg = OmegaConf.create({"MODEL": {"IMAGE_SIZE": args.image_size}})
    renderer = MeshRenderer(cfg, faces=varen.faces)

    for n in parse_samples(args.samples):
        refined_path = os.path.join(args.root, f"equine_out{n}", "postrefine", "refined_results.pt")
        raw_path = os.path.join(args.root, f"equine_out{n}", "animer_outputs.pt")
        if not os.path.exists(refined_path):
            print(f"  sample {n}: no refined_results.pt -- skipped")
            continue
        refined = torch.load(refined_path, map_location='cpu', weights_only=False)
        raw = torch.load(raw_path, map_location='cpu', weights_only=False)

        image = denormalize(raw['img'])
        if args.raw:
            # Pre-refinement AniMer prediction, rebuilt from animer_outputs.pt's
            # rotation matrices, rendered with its own pred_cam_t. Lets the
            # post-optimisation's effect be seen directly rather than inferred.
            from pytorch3d.transforms import matrix_to_axis_angle
            from amr.models.varen_wrapper import VAREN as _V
            go = matrix_to_axis_angle(raw['pred_global_orient'].float().reshape(1, 3, 3)).reshape(1, 3)
            bp = matrix_to_axis_angle(raw['pred_pose'].float().reshape(37, 3, 3)).reshape(1, 111)
            vertices = varen(global_orient=go, body_pose=bp,
                             betas=raw['pred_betas'].float(), transl=torch.zeros(1, 3),
                             pose2rot=True).vertices[0].detach().numpy()
            cam_t = raw['pred_cam_t'][0].float().numpy()
        else:
            vertices = refined['vertices'][0].float().numpy()
            cam_t = refined['refined_cam_t'][0].float().numpy()
        cam_scale = (raw['pred_cam'] if args.raw else refined['refined_cam'])[0, 0]
        f_implied = implied_focal(cam_scale, cam_t[2], args.image_size)

        panels = [image]
        # __call__ mutates camera_translation in place (mesh_renderer.py:245), so
        # every call gets its own copy.
        panels.append(renderer(vertices, cam_t.copy(), image,
                               focal_length=args.focal_length, side_view=False))
        if args.side_view:
            panels.append(renderer(vertices, cam_t.copy(), image,
                                   focal_length=args.focal_length, side_view=True))

        out = os.path.join(out_dir, f"equine_out{n}_overlay{'_raw' if args.raw else ''}.png")
        Image.fromarray((np.concatenate(panels, axis=1) * 255).astype(np.uint8)).save(out)
        print(f"  sample {n}: implied focal {f_implied:8.1f} (using {args.focal_length:.0f}) -> {out}")


if __name__ == "__main__":
    main()
