scripts/train.sh --name horse_v1 -- \
    DATASETS.HORSE.WEIGHT=1 \
    DATASETS.HORSE.ROOT_IMAGE=/lustre/home/okumar/outputs/batches \
    DATASETS.HORSE.JSON_FILE.TRAIN=/lustre/home/okumar/outputs/horse_dataset/train_subset_50.json \
    DATASETS.HORSE.JSON_FILE.TEST=/lustre/home/okumar/outputs/horse_dataset/test_subset_10.json \
    DATASETS.ANIMAL3D.WEIGHT=0 DATASETS.CUB.WEIGHT=0 DATASETS.CTRLAVES3D.WEIGHT=0 \
    MODEL.BACKBONE.PRETRAINED_WEIGHTS=data/AniMerPlus/checkpoint.ckpt \
    MODEL.BACKBONE.FREEZE_ATTN=true MODEL.BACKBONE.FREEZE_FFN=true \
    MODEL.BACKBONE.USE_CLS=false