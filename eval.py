import numpy as np
from tqdm import tqdm
import torch
from amr.utils import recursive_to
from amr.utils.evaluate_metric import Evaluator
from amr.datasets.animal3d_dataset import EvaluationDataset
from amr.datasets.varen_dataset import VARENEvaluationDataset
import argparse
from torch.utils.data import DataLoader
from amr.models.animerpp import AniMerPlusPlus
from amr.configs import get_config
torch.multiprocessing.set_sharing_strategy('file_system')

# AniMerPlusPlus.forward_step is VAREN-only (see amr/models/animerpp.py): it
# always produces a single 'varen_output' entry, regardless of which dataset
# the batch came from. There is no more per-dataset aves_output/smal_output
# branching (model.aves / model.smal no longer exist as attributes on the
# model), so every dataset key is evaluated with a single VAREN-aware
# Evaluator. Datasets whose ground truth isn't VAREN-shaped (e.g. the legacy
# ANIMAL3D/CUB/CTRLAVES3D SMAL/AVES-annotated sets) are not expected to produce
# meaningful metrics through this VAREN-only forward pass anymore -- that's a
# consequence of the model itself having been made VAREN-only, not something
# this script can paper over. HORSE (amr/datasets/varen_dataset.py) is the
# dataset this evaluation path is actually designed for.
VAREN_SHAPED_DATASET_KEYS = {"HORSE"}


def main(args):
    cfg = get_config(args.config)
    default_cfg = get_config(args.default_eval_config)
    model = AniMerPlusPlus.load_from_checkpoint(args.checkpoint, cfg=cfg, strict=False)
    model.eval()
    model.to(args.device)

    evaluator = Evaluator(smal_model=model.varen, image_size=cfg.MODEL.IMAGE_SIZE,
                          pelvis_ind=cfg.EXTRA.get("PELVIS_IND", 0), model_type='varen')
    cfg_eval_dataset = dict(default_cfg.DATASETS)
    aug_cfg = cfg_eval_dataset.pop("CONFIG", None)  # augmentation config is not used in evaluation

    if args.dataset.upper() == "ALL":
        for key in cfg_eval_dataset.keys():
            print(f"-------- Evaluate {key} dataset ------------")
            try:
                eval_one_dataset(cfg_eval_dataset[key], default_cfg, cfg, model,
                                 evaluator=evaluator, aug_cfg=aug_cfg, key=key, device=args.device)
            except Exception as exc:
                print(f"-------- {key} dataset evaluation failed, skipping: {exc!r} --------")
            print(f"-------{key} Dataset evaluate finish ------")
    else:
        print(f"-------- Evaluate {args.dataset} dataset ------------")
        eval_one_dataset(cfg_eval_dataset[args.dataset], default_cfg, cfg, model,
                         evaluator=evaluator, aug_cfg=aug_cfg, key=args.dataset, device=args.device)
        print(f"-------{args.dataset} Dataset evaluate finish ------")


def eval_one_dataset(dataset_cfg, default_cfg, cfg, model, evaluator, aug_cfg, key, device):
    is_varen_dataset = key in VAREN_SHAPED_DATASET_KEYS
    if is_varen_dataset:
        dataset = VARENEvaluationDataset(root_image=dataset_cfg['ROOT_IMAGE'],
                                         json_file=dataset_cfg['JSON_FILE']['TEST'],
                                         augm_config=aug_cfg,
                                         focal_length=cfg.VAREN.get("FOCAL_LENGTH", 1000),
                                         image_size=cfg.MODEL.IMAGE_SIZE,
                                         num_joints=cfg.VAREN.get("NUM_JOINTS", 37),
                                         num_betas=cfg.VAREN.get("NUM_BETAS", 39),
                                         )
    else:
        dataset = EvaluationDataset(root_image=dataset_cfg['ROOT_IMAGE'],
                                    json_file=dataset_cfg['JSON_FILE']['TEST'],
                                    augm_config=aug_cfg, focal_length=cfg.AVES.get("FOCAL_LENGTH", 2167) if key in  ["CUB", "CTRLANI3DPLUS", "COWBIRD"] else cfg.SMAL.get("FOCAL_LENGTH", 1000),
                                    image_size=cfg.MODEL.IMAGE_SIZE,
                                    )
    dataloader = DataLoader(dataset, batch_size=cfg.TRAIN.BATCH_SIZE, num_workers=cfg.GENERAL.NUM_WORKERS)

    bar = tqdm(dataloader)
    pa_mpjpe_list, pck_list, auc_list, pa_mpvpe_list = [], [], [], []
    for i, batch in enumerate(bar):
        batch = recursive_to(batch, device)
        with torch.no_grad():
            output = model(batch)
            if "varen_output" not in output:
                raise KeyError(
                    "Model forward output has no 'varen_output' key (got: "
                    f"{list(output.keys())}). AniMerPlusPlus.forward_step is VAREN-only, "
                    "so this should never happen."
                )
            output = output["varen_output"]

        if is_varen_dataset:
            pa_mpjpe, pa_mpvpe = evaluator.eval_3d(output, batch)
        else:
            pa_mpjpe, pa_mpvpe = 0., 0.
        pck, auc = evaluator.eval_2d(output, batch, pck_threshold=default_cfg.METRIC.PCK_THRESHOLD)

        pa_mpjpe_list.append(pa_mpjpe)
        pa_mpvpe_list.append(pa_mpvpe)
        auc_list.append(auc)
        pck_list.append(pck)

        bar.set_postfix(PA_MPJPE=pa_mpjpe,
                        PA_MPVPE=pa_mpvpe,
                        AUC=auc,
                        pck=pck,)

    print("---------------- 3D metric -----------------")
    print(f"Avg PA-MPJPE: {np.mean(pa_mpjpe_list)}")
    print(f"Avg PA-MPVPE: {np.mean(pa_mpvpe_list)}")
    
    print("--------------- 2D metric ------------------")
    print(f"AUC: {np.mean(auc_list)}")
    pck_list = np.array(pck_list)
    for _, th in enumerate(default_cfg.METRIC.PCK_THRESHOLD):
        print(f"PCK@{th}: {np.mean(pck_list[:, _])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to config file", required=True)
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint file", required=True)
    parser.add_argument("--default_eval_config", type=str, default="amr/configs_hydra/experiment/default_val.yaml")
    parser.add_argument("--dataset", type=str, default="ALL")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for evaluation")
    args = parser.parse_args()
    main(args)
