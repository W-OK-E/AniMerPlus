scripts/train.sh --name horse_orientation-fixed -- \
    DATASETS.HORSE.WEIGHT=1 \
    DATASETS.HORSE.ROOT_IMAGE=/lustre/home/okumar/outputs/batches \
    DATASETS.HORSE.JSON_FILE.TRAIN=/lustre/home/okumar/outputs/horse_dataset/train.json \
    DATASETS.HORSE.JSON_FILE.TEST=/lustre/home/okumar/outputs/horse_dataset/test.json \
    MODEL.BACKBONE.PRETRAINED_WEIGHTS=data/AniMerPlus/checkpoint.ckpt \
    MODEL.BACKBONE.FREEZE_ATTN=false MODEL.BACKBONE.FREEZE_FFN=false \
    MODEL.BACKBONE.FROZEN_STAGES=-1