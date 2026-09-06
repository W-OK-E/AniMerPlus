import pytorch_lightning as pl
import torch
from torch.utils.data import WeightedRandomSampler
from yacs.config import CfgNode

from amr.utils.pylogger import get_pylogger

from .animerpp_dataset import AniMerPlusPlusDataset
from .varen_dataset import VARENTrain3DDataset

log = get_pylogger(__name__)


class AniMerPlusPlusDataModule(pl.LightningDataModule):
    def __init__(self, cfg: CfgNode) -> None:
        """
        Initialize LightningDataModule for AniMerPlusPlus training
        Args:
            cfg (CfgNode): Config file as a yacs CfgNode containing necessary dataset info.
        """
        super().__init__()
        self.cfg = cfg
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.mocap_dataset = None
        self.weight_sampler = None

    def setup(self, stage: str | None = None) -> None:
        """
        Load datasets necessary for training
        Args:
            stage:
        """
        if self.train_dataset is None:
            self.train_dataset = AniMerPlusPlusDataset(self.cfg)
            self.weight_sampler = WeightedRandomSampler(weights=self.train_dataset.weights, 
                                                        num_samples=len(self.train_dataset))
        if self.val_dataset is None:
            # AniMerPlusPlus.forward_step is VAREN-only (see amr/models/animerpp.py):
            # it always predicts the 43 VAREN surface keypoints, regardless of which
            # dataset a batch came from, so validation must use a VAREN-shaped
            # dataset -- SMAL-shaped ground truth (26 keypoints) would fail
            # shape-wise inside compute_varen_loss. The legacy ANIMAL3D fallback
            # that used to live here went away with animal3d_dataset.py.
            dataset_configs = self.cfg.DATASETS
            self.val_dataset = VARENTrain3DDataset(self.cfg, is_train=False,
                                                   root_image=dataset_configs.HORSE.ROOT_IMAGE,
                                                   json_file=dataset_configs.HORSE.JSON_FILE.TEST)

    def train_dataloader(self) -> dict:
        """
        Setup training data loader.
        Returns:
            Dict: Dictionary containing image and dataloaders
        """
        shuffle = not self.weight_sampler is not None
        train_dataloader = torch.utils.data.DataLoader(self.train_dataset, self.cfg.TRAIN.BATCH_SIZE, drop_last=True,
                                                       num_workers=self.cfg.GENERAL.NUM_WORKERS,
                                                       prefetch_factor=self.cfg.GENERAL.PREFETCH_FACTOR,
                                                       pin_memory=True,
                                                       shuffle=shuffle,
                                                       sampler=self.weight_sampler,
                                                       )
        return {'img': train_dataloader}

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Setup val data loader.
        Returns:
            torch.utils.data.DataLoader: Validation dataloader
        """
        val_dataloader = torch.utils.data.DataLoader(self.val_dataset, self.cfg.TRAIN.BATCH_SIZE, drop_last=True,
                                                     num_workers=self.cfg.GENERAL.NUM_WORKERS, pin_memory=True)
        return val_dataloader
    