from .vit_moe import vithmoe


def create_backbone(cfg):
    if cfg.MODEL.BACKBONE.TYPE == 'vithmoe':
        return vithmoe(cfg)
    else:
        raise NotImplementedError('Backbone type is not implemented')
