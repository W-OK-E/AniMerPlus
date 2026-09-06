import torch
from yacs.config import CfgNode

from ..utils.pylogger import get_pylogger
from .varen_dataset import VARENTrain3DDataset

log = get_pylogger(__name__)


class AniMerPlusPlusDataset(torch.utils.data.Dataset):
    def __init__(self, cfg: CfgNode):
        """This abstract class exists so that if and when we are creating datasets for other animals
        or species, we can just add them here and concatenate them together."""
        dataset_configs = cfg.DATASETS
        self.dataset = None
        self.dataset = VARENTrain3DDataset(cfg, is_train=True,
                                                    root_image=dataset_configs.HORSE.ROOT_IMAGE,
                                                    json_file=dataset_configs.HORSE.JSON_FILE.TRAIN)
        log.info("HORSE Dataset loading finished")        
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]
