#!/usr/bin/env python
"""
Compare AniMer, 4DEquine and this pipeline against the synthetic VAREN ground truth.

WHAT MAKES THIS COMPARABLE
--------------------------
Every mesh is built with transl=0, so global_orient places GT and prediction in
one common model frame. Orientation is therefore *not* a free parameter: the
unaligned per-vertex error below legitimately contains orientation, articulation
and size error, and no camera is needed (we compare no 2D projections).

Three things had to be equalised first, each verified rather than assumed:

1. use_muscle_deformations=False. 4DEquine's stored vertices match its own
   params to 0.0016 mean with muscle deformations OFF and only 0.017 with them
   ON, and amr's wrapper (animerpp.py:70) also defaults to False. The variant
   difference is the same order as real prediction error, so mixing them would
   swamp the comparison.

2. Tail exclusion. 4DEquine additionally predicts `refined_tail_scale`, which
   the VAREN params schema cannot express. The affected vertices are derived
   here at runtime (stored-vs-rebuilt, over every sample) rather than hardcoded,
   and are dropped from EVERY model so all share one vertex subset.

3. AniMer's frame. Its .obj is SMAL topology in render coordinates:
   Animer-latest/amr/utils/renderer.py:245-254 builds it as
   (pred_vertices + cam_t) rotated 180 deg about X, i.e. * [1,-1,-1]. That
   rotation is inverted exactly here; cam_t is not saved but is a pure
   translation, so the centroid-aligned Chamfer is unaffected.

AniMer needs Chamfer because SMAL (3889 verts) has no vertex correspondence
with VAREN (13873). It is reported twice -- raw, and with its scale normalised
to GT -- because its size ratio is a near-constant ~0.69, which is more likely a
systematic SMAL-vs-VAREN template offset than per-sample error. Chamfer is
reported for all three so there is one column that is like-for-like.

USAGE:
    python scripts/compare_pipelines.py --root /home/om/mpi/symp/test_data/symp_viz_dat
"""
import argparse
import json
import os

import numpy as np
import pyrootutils
import torch

root = pyrootutils.setup_root(
    search_from=__file__, indicator=[".git", "pyproject.toml"], pythonpath=True, dotenv=True)

NUM_JOINTS = 37


def to_camera_frame(v: np.ndarray) -> np.ndarray:
    """VAREN native -> OpenCV camera frame, matching animerpp._varen_native_to_camera_frame."""
    return np.stack([v[:, 0], -v[:, 2], v[:, 1]], axis=1)


# 4DEquine expresses its params in the CAMERA frame, while the dataset GT (and
# this pipeline's predictions) are in VAREN's native frame. Evidence, printed at
# runtime by --show-frame-diagnostic: uncorrected, 4DEquine's global_orient error
# is 78-93 deg on every one of the 10 samples -- clustered at ~90, the signature
# of a constant frame offset rather than prediction error -- and drops to 14-28
# deg once corrected. The correction is exactly Rx(-90), the inverse of
# animerpp._varen_native_to_camera_frame, and applying it to *our* predictions
# makes them 4x worse (227 -> 917 mm), confirming it is specific to 4DEquine.
# Left uncorrected, ~64 deg of pure convention would be charged to 4DEquine.
R_CAMERA_TO_NATIVE = np.array([[1.0, 0.0, 0.0],
                               [0.0, 0.0, 1.0],
                               [0.0, -1.0, 0.0]])


def correct_frame(params: dict) -> dict:
    """Re-express params whose global_orient is camera-frame into native frame.

    Applied to global_orient rather than to the vertices: with transl=0 the root
    joint sits at the origin so the two are numerically identical (verified), but
    doing it in params keeps the rotation metrics and the mesh metrics consistent.
    """
    from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
    R = axis_angle_to_matrix(torch.tensor(params['global_orient'], dtype=torch.float32).reshape(1, 3))[0]
    fixed = matrix_to_axis_angle((torch.tensor(R_CAMERA_TO_NATIVE, dtype=torch.float32) @ R).unsqueeze(0))[0]
    return {**params, 'global_orient': fixed.numpy().tolist()}


def undo_render_rotation(v: np.ndarray) -> np.ndarray:
    """Invert the 180-degree-about-X that AniMer's renderer bakes into its .obj."""
    return v * np.array([1.0, -1.0, -1.0])


def chamfer(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric mean nearest-neighbour distance; topology-agnostic."""
    from scipy.spatial import cKDTree
    da, _ = cKDTree(b).query(a)
    db, _ = cKDTree(a).query(b)
    return 0.5 * (da.mean() + db.mean())


def procrustes_align(S1: np.ndarray, S2: np.ndarray) -> np.ndarray:
    """Rotation+scale+translation of S1 onto S2; requires correspondence."""
    mu1, mu2 = S1.mean(0), S2.mean(0)
    X1, X2 = S1 - mu1, S2 - mu2
    K = X1.T @ X2
    U, _, Vt = np.linalg.svd(K)
    Z = np.eye(3)
    Z[-1, -1] = np.sign(np.linalg.det(U @ Vt))
    R = Vt.T @ Z @ U.T
    scale = np.trace(R @ K) / (X1 ** 2).sum()
    return scale * (R @ X1.T).T + mu2


def geodesic_deg(aa_a: np.ndarray, aa_b: np.ndarray) -> np.ndarray:
    """Per-rotation geodesic angle in degrees between two (N,3) axis-angle sets."""
    from pytorch3d.transforms import axis_angle_to_matrix
    Ra = axis_angle_to_matrix(torch.tensor(aa_a, dtype=torch.float64).reshape(-1, 3))
    Rb = axis_angle_to_matrix(torch.tensor(aa_b, dtype=torch.float64).reshape(-1, 3))
    trace = torch.einsum('nij,nij->n', Ra, Rb)          # trace(Ra^T Rb)
    cos = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cos)).numpy()


class MeshBuilder:
    """One VAREN instance, so every model goes through an identical params->mesh path."""

    def __init__(self, model_path: str, num_betas: int = 39):
        from varen import VAREN
        self.model = VAREN(model_path, use_muscle_deformations=False)
        self.num_betas = num_betas

    def build(self, global_orient, pose, betas) -> np.ndarray:
        out = self.model(
            global_orient=torch.tensor(global_orient, dtype=torch.float32).reshape(1, 3),
            body_pose=torch.tensor(pose, dtype=torch.float32).reshape(1, NUM_JOINTS * 3),
            betas=torch.tensor(betas, dtype=torch.float32).reshape(1, self.num_betas),
            transl=torch.zeros(1, 3))
        return out.vertices.squeeze(0).detach().numpy()


def derive_tail_mask(root_dir: str, samples: list[int], builder: MeshBuilder,
                     threshold: float = 0.005) -> np.ndarray:
    """Vertices 4DEquine's refined_tail_scale moves, found by comparing its stored
    mesh against one rebuilt from the params it also stores."""
    hits = None
    for n in samples:
        path = os.path.join(root_dir, f"equine_out{n}", "postrefine", "refined_results.pt")
        if not os.path.exists(path):
            continue
        r = torch.load(path, map_location='cpu', weights_only=False)
        p = load_equine_params(root_dir, n)
        v = builder.build(p['global_orient'], p['pose'], p['betas'])
        d = np.linalg.norm(v - r['vertices'][0].float().numpy(), axis=1)
        hits = (d > threshold) if hits is None else (hits | (d > threshold))
    return hits


def load_gt(root_dir: str, n: int) -> dict:
    with open(os.path.join(root_dir, f"sample{n}", "metadata.json")) as f:
        md = json.load(f)
    return {'global_orient': md['pose'][:3], 'pose': md['pose'][3:], 'betas': md['shape'],
            'img_path': md['img_path']}


def load_equine_params(root_dir: str, n: int) -> dict:
    with open(os.path.join(root_dir, f"equine_out{n}", "params.json")) as f:
        return json.load(f)


def load_equine_raw_params(root_dir: str, n: int) -> dict:
    """4DEquine's PRE-refinement prediction, straight out of animer_outputs.pt.

    Reported as its own column because post-refinement is what wrecks this
    method on single images: it fits pose/orient/betas/cam to ViTPose keypoints
    at w_keypoints=10000 (post_optimization_from_video.py:148), and on this data
    only 3 of 170 ViTPose keypoints land on the animal at all. The raw regressor
    is strong; the refinement stage is what fails, so showing both separates
    "the method is bad" from "one stage ran outside its design envelope".
    """
    from pytorch3d.transforms import matrix_to_axis_angle
    a = torch.load(os.path.join(root_dir, f"equine_out{n}", "animer_outputs.pt"),
                   map_location='cpu', weights_only=False)
    return {
        'global_orient': matrix_to_axis_angle(
            a['pred_global_orient'].float().reshape(1, 3, 3)).reshape(-1).tolist(),
        'pose': matrix_to_axis_angle(
            a['pred_pose'].float().reshape(NUM_JOINTS, 3, 3)).reshape(-1).tolist(),
        'betas': a['pred_betas'][0].float().tolist(),
    }


def load_ours_params(root_dir: str, n: int) -> dict:
    with open(os.path.join(root_dir, "ours_output", f"holdout_{n-1:03d}_pred_params.json")) as f:
        return json.load(f)


def load_animer_obj(root_dir: str, n: int):
    import trimesh
    path = os.path.join(root_dir, f"animer_out{n}", "img_path_0.obj")
    if not os.path.exists(path):
        return None
    return np.asarray(trimesh.load(path, process=False).vertices)


def param_metrics(gt: dict, pred: dict) -> dict:
    """Only possible because both sides expose VAREN params, not just a mesh."""
    return {
        'global_orient_deg': float(geodesic_deg(np.array(gt['global_orient']),
                                                np.array(pred['global_orient']))[0]),
        'joint_rot_deg': float(geodesic_deg(np.array(gt['pose']).reshape(-1, 3),
                                            np.array(pred['pose']).reshape(-1, 3)).mean()),
        'betas_l2': float(np.linalg.norm(np.array(gt['betas']) - np.array(pred['betas']))),
    }


def mesh_metrics(gt_v: np.ndarray, pred_v: np.ndarray, keep: np.ndarray, scale_mm: float) -> dict:
    g, p = gt_v[keep], pred_v[keep]
    return {
        'mpvpe_mm': float(np.linalg.norm(p - g, axis=1).mean() * scale_mm),
        'pa_mpvpe_mm': float(np.linalg.norm(procrustes_align(p, g) - g, axis=1).mean() * scale_mm),
    }


def summarize(rows: list[dict], keys: list[str]) -> dict:
    out = {}
    for k in keys:
        vals = [r[k] for r in rows if r.get(k) is not None]
        out[k] = float(np.mean(vals)) if vals else None
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="symp_viz_dat directory")
    p.add_argument("--samples", default="1-10")
    p.add_argument("--varen-model-path", default="/home/om/mpi/VAREN/models")
    p.add_argument("--scale-mm", type=float, default=1000.0,
                   help="model units -> mm (VAREN units are metric, so 1000)")
    p.add_argument("--out-csv", default=None, help="default: <root>/metrics_per_sample.csv")
    p.add_argument("--no-equine-frame-fix", action="store_true",
                   help="report 4DEquine in its raw camera frame (see R_CAMERA_TO_NATIVE); "
                        "this charges it ~64 deg of pure convention and is only useful "
                        "for reproducing the uncorrected numbers")
    args = p.parse_args()

    lo, hi = args.samples.split('-')
    samples = list(range(int(lo), int(hi) + 1))
    builder = MeshBuilder(args.varen_model_path)

    # Frame diagnostic, printed so the correction below is auditable rather than
    # a silent transform buried in the numbers.
    raw, fixed = [], []
    for n in samples:
        gt = load_gt(args.root, n)
        ep = load_equine_params(args.root, n)
        raw.append(geodesic_deg(np.array(gt['global_orient']), np.array(ep['global_orient']))[0])
        fixed.append(geodesic_deg(np.array(gt['global_orient']),
                                  np.array(correct_frame(ep)['global_orient']))[0])
    print(f"4DEquine global_orient vs GT: raw {np.mean(raw):.1f} deg "
          f"(range {np.min(raw):.0f}-{np.max(raw):.0f}), "
          f"frame-corrected {np.mean(fixed):.1f} deg "
          f"(range {np.min(fixed):.0f}-{np.max(fixed):.0f})")
    print("  -> raw errors cluster at ~90 deg on every sample = constant frame offset, "
          f"{'CORRECTION APPLIED' if not args.no_equine_frame_fix else 'CORRECTION DISABLED'}\n")

    keep = ~derive_tail_mask(args.root, samples, builder)
    print(f"tail-scale vertices excluded from every model: {(~keep).sum()} / {keep.size} "
          f"({100*(~keep).sum()/keep.size:.2f}%)\n")

    rows = []
    for n in samples:
        gt = load_gt(args.root, n)
        gt_v = builder.build(gt['global_orient'], gt['pose'], gt['betas'])
        gt_cam = to_camera_frame(gt_v)
        body_len = np.ptp(gt_v, axis=0).max()
        row = {'sample': n, 'img': gt['img_path']}

        equine_params = load_equine_params(args.root, n)
        equine_raw_params = load_equine_raw_params(args.root, n)
        if not args.no_equine_frame_fix:
            equine_params = correct_frame(equine_params)
            equine_raw_params = correct_frame(equine_raw_params)
        for tag, params in (('equine_raw', equine_raw_params),
                            ('equine', equine_params),
                            ('ours', load_ours_params(args.root, n))):
            pv = builder.build(params['global_orient'], params['pose'], params['betas'])
            row.update({f'{tag}_{k}': v for k, v in mesh_metrics(gt_v, pv, keep, args.scale_mm).items()})
            row.update({f'{tag}_{k}': v for k, v in param_metrics(gt, params).items()})
            pc = to_camera_frame(pv)[keep]
            row[f'{tag}_chamfer_mm'] = chamfer(pc - pc.mean(0) + gt_cam[keep].mean(0),
                                               gt_cam[keep]) * args.scale_mm

        av = load_animer_obj(args.root, n)
        if av is None:
            row['animer_detected'] = 0
            for k in ('animer_chamfer_mm', 'animer_chamfer_scaled_mm', 'animer_size_ratio'):
                row[k] = None
        else:
            row['animer_detected'] = 1
            a = undo_render_rotation(av)
            ratio = np.ptp(a, axis=0).max() / np.ptp(gt_cam, axis=0).max()
            row['animer_size_ratio'] = float(ratio)
            ac = a - a.mean(0) + gt_cam.mean(0)
            row['animer_chamfer_mm'] = chamfer(ac, gt_cam) * args.scale_mm
            s = (a - a.mean(0)) / ratio + gt_cam.mean(0)
            row['animer_chamfer_scaled_mm'] = chamfer(s, gt_cam) * args.scale_mm
        row['gt_body_len'] = float(body_len)
        rows.append(row)

    out_csv = args.out_csv or os.path.join(args.root, "metrics_per_sample.csv")
    import csv
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    det = [r for r in rows if r['animer_detected']]
    print(f"AniMer detection: {len(det)}/{len(rows)} samples "
          f"(failed on {[r['sample'] for r in rows if not r['animer_detected']]})\n")

    mesh_keys = ['mpvpe_mm', 'pa_mpvpe_mm', 'chamfer_mm']
    par_keys = ['global_orient_deg', 'joint_rot_deg', 'betas_l2']

    tags = ('equine_raw', 'equine', 'ours')
    headers = ('4DEq raw', '4DEq refined', 'Ours')

    def table(title, subset):
        print(f"=== {title} (n={len(subset)}) ===")
        print(f"{'metric':<22}" + "".join(f"{h:>14}" for h in headers) + f"{'AniMer':>12}")
        s = {t: summarize(subset, [f'{t}_{k}' for k in mesh_keys + par_keys]) for t in tags}
        for k, label in zip(mesh_keys + par_keys,
                            ['MPVPE (mm)', 'PA-MPVPE (mm)', 'Chamfer (mm)',
                             'global_orient (deg)', 'mean joint rot (deg)', 'betas L2']):
            cells = "".join(f"{s[t][f'{t}_{k}']:14.1f}" for t in tags)
            a = summarize(subset, ['animer_chamfer_mm'])['animer_chamfer_mm'] if k == 'chamfer_mm' else None
            cells += f"{a:12.1f}" if a is not None else f"{'n/a':>12}"
            print(f"{label:<22}{cells}")
        a_s = summarize(subset, ['animer_chamfer_scaled_mm', 'animer_size_ratio'])
        if a_s['animer_chamfer_scaled_mm'] is not None:
            pad = "".join(f"{'-':>14}" for _ in tags)
            print(f"{'Chamfer scale-norm (mm)':<22}{pad}{a_s['animer_chamfer_scaled_mm']:12.1f}")
            print(f"{'AniMer size / GT':<22}{pad}{a_s['animer_size_ratio']:12.3f}")
        print()

    table("COMMON SAMPLES (AniMer detected)", det)
    table("ALL SAMPLES", rows)
    print(f"per-sample CSV -> {out_csv}")


if __name__ == "__main__":
    main()
