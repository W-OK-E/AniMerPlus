scripts/train.sh --name horse_orientation-fixed -- \
    DATASETS.HORSE.WEIGHT=1 \
    DATASETS.HORSE.ROOT_IMAGE=/lustre/home/okumar/outputs/batches \
    DATASETS.HORSE.JSON_FILE.TRAIN=/lustre/home/okumar/outputs/horse_dataset/train.json \
    DATASETS.HORSE.JSON_FILE.TEST=/lustre/home/okumar/outputs/horse_dataset/test.json \
    DATASETS.ANIMAL3D.WEIGHT=0 DATASETS.CUB.WEIGHT=0 DATASETS.CTRLAVES3D.WEIGHT=0 \
    MODEL.BACKBONE.PRETRAINED_WEIGHTS=data/AniMerPlus/checkpoint.ckpt \
    MODEL.BACKBONE.FREEZE_ATTN=false MODEL.BACKBONE.FREEZE_FFN=false \
    MODEL.BACKBONE.FROZEN_STAGES=27 \
    MODEL.BACKBONE.USE_CLS=false