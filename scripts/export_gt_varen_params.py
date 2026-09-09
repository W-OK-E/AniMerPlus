#!/usr/bin/env python
"""
Export ground-truth VAREN params from a rendered-dataset JSON.

Companion to export_varen_params.py, which does the same job for 4DEquine's
.pt outputs. This one reads the synthetic dataset's own annotations, so the
result is the GT to measure every pipeline against.

Output schema matches the layout already in use in symp_img1/params.json --
which is example_params.json's three keys plus the extras needed for metrics
that example_params.json cannot express:

    global_orient  (3,)    = dataset 'pose'[:3]      axis-angle
    pose           (111,)  = dataset 'pose'[3:]      axis-angle, 37 joints
    betas          (39,)   = dataset 'shape'
    trans          (3,)    = dataset 'trans'         metric camera translation
    bbox           (4,)    x, y, w, h
    img_path / mask_path   as recorded in the dataset

VAREN/examples/visualise_model.py reads only global_orient/pose/betas and
ignores the rest, so these files feed it directly.

USAGE:
    # the two presentation samples
    python scripts/export_gt_varen_params.py symp_img2/dataset_textured.json \
        --images sample_0022.png sample_0027.png --out-dir symp_img2

    # one dir per sample, so `visualise_model.py --output_path <dir>` can drop
    # its VAREN_full.ply beside the params (the symp_img1 layout)
    python scripts/export_gt_varen_params.py symp_img2/dataset_textured.json \
        --images sample_0022.png sample_0027.png --out-dir symp_img2 --per-sample-dirs
"""
import argparse
import json
import os

NUM_JOINTS = 37
NUM_BETAS = 39


def find_sample(data: list, image: str) -> tuple[int, dict]:
    """Locate the one entry whose img_path ends with `image`.

    Ambiguity is an error rather than a silent first-match: dataset img_paths
    are often batch-prefixed (`batch_0044/images_textured/sample_0029.png`), so
    a bare filename can legitimately name several samples across batches.
    """
    hits = [(i, s) for i, s in enumerate(data) if s['img_path'].endswith(image)]
    if not hits:
        raise KeyError(f"no dataset entry matches image {image!r}")
    if len(hits) > 1:
        paths = [s['img_path'] for _, s in hits]
        raise KeyError(
            f"image {image!r} matches {len(hits)} entries: {paths}. "
            f"Pass a longer suffix (e.g. 'batch_0003/images_textured/{image}') to disambiguate.")
    return hits[0]


def sample_to_params(sample: dict) -> dict:
    """Split the dataset's flat 114-value 'pose' into VAREN's global_orient + pose."""
    pose = sample['pose']
    expected = 3 * (NUM_JOINTS + 1)
    if len(pose) != expected:
        raise ValueError(
            f"'pose' has {len(pose)} values, expected {expected} "
            f"(3 for global_orient + {NUM_JOINTS}x3 joints)")
    betas = sample['shape']
    if len(betas) != NUM_BETAS:
        raise ValueError(f"'shape' has {len(betas)} values, expected {NUM_BETAS}")
    return {
        'img_path': sample['img_path'],
        'mask_path': sample.get('mask_path'),
        'bbox': sample['bbox'],
        'pose': pose[3:],
        'betas': betas,
        'trans': sample['trans'],
        'global_orient': pose[:3],
    }


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset_json", help="the rendered dataset's JSON (holds 'data' and 'camera')")
    p.add_argument("--images", nargs='+', required=True,
                   help="image filenames (or path suffixes) to export")
    p.add_argument("--out-dir", default=".", help="where to write the params files")
    p.add_argument("--per-sample-dirs", action="store_true",
                   help="write <out-dir>/<stem>/params.json instead of "
                        "<out-dir>/<stem>_params.json, so each sample has its own "
                        "folder for visualise_model.py's VAREN_full.ply")
    p.add_argument("--indent", type=int, default=None,
                   help="JSON indent; omit for the compact single-line form")
    args = p.parse_args()

    with open(args.dataset_json) as f:
        dataset = json.load(f)
    data = dataset['data']
    print(f"{args.dataset_json}: {len(data)} samples, camera={dataset.get('camera')}")

    for image in args.images:
        idx, sample = find_sample(data, image)
        params = sample_to_params(sample)
        stem = os.path.splitext(os.path.basename(sample['img_path']))[0]
        if args.per_sample_dirs:
            out = os.path.join(args.out_dir, stem, "params.json")
        else:
            out = os.path.join(args.out_dir, f"{stem}_params.json")
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, 'w') as f:
            json.dump(params, f, indent=args.indent)
        print(f"  [{idx}] {sample['img_path']}  texture={sample.get('texture')}  -> {out}")


if __name__ == "__main__":
    main()
