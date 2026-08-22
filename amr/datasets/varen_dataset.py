"""
VAREN-shaped dataset loaders for horse data.
"""
import copy
import os
import numpy as np
from yacs.config import CfgNode
import cv2
import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

import json
from PIL import Image
from typing import List
from torch.utils.data import Dataset
from .utils import get_example, expand_to_aspect_ratio

HORSE_CATEGORY_ID = 1000
HORSE_SUPERCATEGORY_ID = 1000


def _load_varen_item(data: dict, root_image: str, focal_length: float,
                     num_joints: int, num_betas: int,
                     img_size: int, mean: np.ndarray, std: np.ndarray,
                     is_train: bool, augm_config: CfgNode,
                     use_skimage_antialias: bool, border_mode: int,
                     expand_bbox: bool) -> dict:
    """Shared __getitem__ body for VARENTrain3DDataset / VARENEvaluationDataset."""
    key = data['img_path']
    image = np.array(Image.open(os.path.join(root_image, key)).convert("RGB"))
    mask = np.array(Image.open(os.path.join(root_image, data['mask_path'])).convert('L'))

    category_idx = int(data.get('category', HORSE_CATEGORY_ID))
    supercategory_idx = int(data.get('supercategory', HORSE_SUPERCATEGORY_ID))

    keypoint_2d = np.array(data['keypoint_2d'], dtype=np.float32)  # [43, 3] (x, y, vis)
    # The dataset's exported frame (+Y up) and the pipeline's camera frame (+Y
    # down, +Z depth) differ by exactly a 180-degree rotation about X, hence correction needed
    keypoint_3d_xyz = np.array(data['keypoint_3d'], dtype=np.float32) * np.array([1., -1., -1.], dtype=np.float32)
    keypoint_3d = np.concatenate(
        (keypoint_3d_xyz,
         np.ones((len(data['keypoint_3d']), 1), dtype=np.float32)), axis=-1)  # [43, 4] (x, y, z, conf)

    bbox = data['bbox']  # [x, y, w, h]
    center = np.array([(bbox[0] * 2 + bbox[2]) // 2, (bbox[1] * 2 + bbox[3]) // 2])
    center_x, center_y = center[0], center[1]

    if expand_bbox:
        scale = np.array([bbox[2], bbox[3]], dtype=np.float32) / 200.
        bbox_size = expand_to_aspect_ratio(scale * 200, None).max()
        bbox_expand_factor = bbox_size / ((scale * 200).max())
    else:
        bbox_size = max([bbox[2], bbox[3]])
        bbox_expand_factor = 1.0

    pose = np.array(data['pose'], dtype=np.float32)  # [3 + num_joints*3]
    expected_pose_len = 3 + num_joints * 3
    if pose.shape[0] != expected_pose_len:
        raise ValueError(
            f"VAREN dataset item {key!r}: 'pose' has length {pose.shape[0]}, "
            f"expected {expected_pose_len} (= 3 global_orient + {num_joints}*3 body_pose)."
        )
    betas = np.array(data['shape'], dtype=np.float32)  # [num_betas]
    if betas.shape[0] != num_betas:
        raise ValueError(
            f"VAREN dataset item {key!r}: 'shape' has length {betas.shape[0]}, expected {num_betas}."
        )
    translation = np.array(data['trans'], dtype=np.float32)  # [3]
    has_pose = np.array(1., dtype=np.float32)
    has_betas = np.array(1., dtype=np.float32)
    has_translation = np.array(1., dtype=np.float32)
    ori_keypoint_2d = keypoint_2d.copy()

    aug_params = {'global_orient': pose[:3],
                 'pose': pose[3:],
                 'betas': betas,
                 'transl': translation,
                 }
    has_aug_params = {'global_orient': has_pose,
                      'pose': has_pose,
                      'betas': has_betas,
                      'transl': has_translation,
                      }

    augm_config = copy.deepcopy(augm_config)
    img_rgba = np.concatenate([image, mask[:, :, None]], axis=2)
    img_patch_rgba, keypoints_2d, keypoints_3d, aug_params, has_aug_params, img_size_out, trans, img_border_mask = get_example(
        img_rgba,
        center_x, center_y,
        bbox_size, bbox_size,
        keypoint_2d, keypoint_3d,
        aug_params, has_aug_params,
        img_size, img_size,
        mean, std, is_train, augm_config,
        is_bgr=False, return_trans=True,
        use_skimage_antialias=use_skimage_antialias,
        border_mode=border_mode,
    )
    img_patch = (img_patch_rgba[:3, :, :])
    mask_patch = (img_patch_rgba[3, :, :] / 255.0).clip(0, 1)
    if (mask_patch < 0.5).all():
        mask_patch = np.ones_like(mask_patch)

    varen_params = {'global_orient': aug_params['global_orient'],
                    'body_pose': aug_params['pose'],
                    'betas': aug_params['betas'],
                    'transl': aug_params['transl'],
                    }
    has_varen_params = {'global_orient': has_aug_params['global_orient'],
                        'body_pose': has_aug_params['pose'],
                        'betas': has_aug_params['betas'],
                        'transl': has_aug_params['transl'],
                        }
    varen_params_is_axis_angle = {'global_orient': True,
                                  'body_pose': True,
                                  'betas': False,
                                  'transl': False,
                                  }

    item = {'img': img_patch,
            'mask': mask_patch,
            'keypoints_2d': keypoints_2d,
            'keypoints_3d': keypoints_3d,
            'orig_keypoints_2d': ori_keypoint_2d,
            'box_center': np.array(center.copy(), dtype=np.float32),
            'box_size': float(bbox_size),
            'img_size': np.array(1.0 * img_size_out[::-1].copy(), dtype=np.float32),
            'varen_params': varen_params,
            'has_varen_params': has_varen_params,
            'varen_params_is_axis_angle': varen_params_is_axis_angle,
            '_trans': trans,
            'focal_length': np.array([focal_length, focal_length], dtype=np.float32),
            'category': np.array(category_idx, dtype=np.int32),
            'supercategory': np.array(supercategory_idx, dtype=np.int32),
            'bbox_expand_factor': bbox_expand_factor,
            'img_border_mask': img_border_mask.astype(np.float32),
            'has_mask': np.array(1, dtype=np.float32)}
    return item


class VARENTrain3DDataset(Dataset):
    """Training-time horse dataset producing VAREN-shaped ground truth."""

    def __init__(self, cfg: CfgNode, is_train: bool, root_image: str, json_file: str):
        super().__init__()
        self.root_image = root_image
        self.focal_length = cfg.VAREN.get("FOCAL_LENGTH", 1000)
        self.num_joints = cfg.VAREN.get("NUM_JOINTS", 37)
        self.num_betas = cfg.VAREN.get("NUM_BETAS", 39)

        with open(json_file, 'r') as f:
            self.data = json.load(f)

        self.is_train = is_train
        self.IMG_SIZE = cfg.MODEL.IMAGE_SIZE
        self.MEAN = 255. * np.array(cfg.MODEL.IMAGE_MEAN)
        self.STD = 255. * np.array(cfg.MODEL.IMAGE_STD)
        self.use_skimage_antialias = cfg.DATASETS.get('USE_SKIMAGE_ANTIALIAS', False)
        self.border_mode = {
            'constant': cv2.BORDER_CONSTANT,
            'replicate': cv2.BORDER_REPLICATE,
        }[cfg.DATASETS.get('BORDER_MODE', 'constant')]

        self.augm_config = cfg.DATASETS.CONFIG

    def __len__(self):
        return len(self.data['data'])

    def __getitem__(self, item):
        data = self.data['data'][item]
        return _load_varen_item(data, self.root_image, self.focal_length,
                                self.num_joints, self.num_betas,
                                self.IMG_SIZE, self.MEAN, self.STD,
                                self.is_train, self.augm_config,
                                self.use_skimage_antialias, self.border_mode,
                                expand_bbox=False)


class VARENEvaluationDataset(Dataset):
    """Eval-time horse dataset producing VAREN-shaped ground truth (no augmentation)."""

    def __init__(self, root_image: str, json_file: str, augm_config,
                focal_length: int = 1000, image_size: int = 256,
                num_joints: int = 37, num_betas: int = 39,
                mean: List[float] = [0.485, 0.456, 0.406], std: List[float] = [0.229, 0.224, 0.225]):
        super().__init__()
        self.root_image = root_image
        self.focal_length = focal_length
        self.num_joints = num_joints
        self.num_betas = num_betas

        with open(json_file, 'r') as f:
            self.data = json.load(f)

        self.is_train = False
        self.IMG_SIZE = image_size
        self.MEAN = 255. * np.array(mean)
        self.STD = 255. * np.array(std)
        self.use_skimage_antialias = False
        self.border_mode = cv2.BORDER_CONSTANT
        self.augm_config = augm_config

    def __len__(self):
        return len(self.data['data'])

    def __getitem__(self, item):
        data = self.data['data'][item]
        return _load_varen_item(data, self.root_image, self.focal_length,
                                self.num_joints, self.num_betas,
                                self.IMG_SIZE, self.MEAN, self.STD,
                                self.is_train, self.augm_config,
                                self.use_skimage_antialias, self.border_mode,
                                expand_bbox=True)
