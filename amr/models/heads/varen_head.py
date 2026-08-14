import torch
import torch.nn as nn
import einops
from ..components.pose_transformer import TransformerDecoder
from ...utils.geometry import rot6d_to_rotmat


def build_varen_head(cfg):
    varen_head_type = cfg.MODEL.VAREN_HEAD.get('TYPE', 'transformer_decoder')
    if varen_head_type == 'transformer_decoder':
        return VARENTransformerDecoderHead(cfg)
    raise ValueError('Unknown VAREN head type: {}'.format(varen_head_type))


class VARENTransformerDecoderHead(nn.Module):
    """Cross-attention based VAREN Transformer decoder."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.joint_rep_type = cfg.MODEL.VAREN_HEAD.get('JOINT_REP', '6d')
        if self.joint_rep_type not in ('6d', 'aa'):
            raise ValueError('Unknown VAREN head joint representation: {}'.format(self.joint_rep_type))
        self.joint_rep_dim = {'6d': 6, 'aa': 3}[self.joint_rep_type]
        npose = self.joint_rep_dim * (cfg.VAREN.NUM_JOINTS + 1)
        self.input_is_mean_shape = cfg.MODEL.VAREN_HEAD.get('TRANSFORMER_INPUT', 'zero') == 'mean_shape'
        transformer_args = dict(
            num_tokens=1,
            token_dim=(npose + cfg.VAREN.get('NUM_BETAS', 39) + 3) if self.input_is_mean_shape else 1,
            dim=1024,
        )
        transformer_args = {**transformer_args, **dict(cfg.MODEL.VAREN_HEAD.TRANSFORMER_DECODER)}

        self.transformer = TransformerDecoder(**transformer_args)
        dim = transformer_args['dim']
        self.decpose = nn.Linear(dim, npose)
        self.decshape = nn.Linear(dim, cfg.VAREN.get('NUM_BETAS', 39))
        self.deccam = nn.Linear(dim, 3)

        if cfg.MODEL.VAREN_HEAD.get('INIT_DECODER_XAVIER', False):
            nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
            nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
            nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        init_pose = torch.zeros(size=(1, npose), dtype=torch.float32)
        init_betas = torch.zeros(size=(1, cfg.VAREN.get('NUM_BETAS', 39)), dtype=torch.float32)
        init_cam = torch.tensor([[4.5, 0, 0]], dtype=torch.float32)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_betas', init_betas)
        self.register_buffer('init_cam', init_cam)

    def forward(self, x, **kwargs):
        batch_size = x.shape[0]
        x = einops.rearrange(x, 'b c h w -> b (h w) c')

        init_pose = self.init_pose.expand(batch_size, -1)
        init_betas = self.init_betas.expand(batch_size, -1)
        init_cam = self.init_cam.expand(batch_size, -1)

        pred_pose = init_pose
        pred_betas = init_betas
        pred_cam = init_cam
        for i in range(self.cfg.MODEL.VAREN_HEAD.get('IEF_ITERS', 3)):
            if self.input_is_mean_shape:
                token = torch.cat([pred_pose, pred_betas, pred_cam], dim=1)[:, None, :]
            else:
                token = torch.zeros(batch_size, 1, 1, device=x.device)

            token_out = self.transformer(token, context=x)
            token_out = token_out.squeeze(1)

            pred_pose = self.decpose(token_out) + pred_pose
            pred_betas = self.decshape(token_out) + pred_betas
            pred_cam = self.deccam(token_out) + pred_cam

        num_joints = self.cfg.VAREN.NUM_JOINTS + 1
        pred_pose = pred_pose.view(batch_size, num_joints, self.joint_rep_dim)
        if self.joint_rep_type == '6d':
            # continuous rotation representation -> rotation matrices (avoids
            # the raw axis-angle discontinuity -- see .agent/Checks.md,
            # "rotation representation" finding). Same conversion smal_head.py
            # already uses for its own (working) 6d path.
            pred_pose = rot6d_to_rotmat(pred_pose.reshape(-1, 6)).view(batch_size, num_joints, 3, 3)
            global_orient = pred_pose[:, [0]]  # (B, 1, 3, 3)
            body_pose = pred_pose[:, 1:]  # (B, NUM_JOINTS, 3, 3)
        else:
            global_orient = pred_pose[:, 0]  # (B, 3)
            body_pose = pred_pose[:, 1:].reshape(batch_size, -1)  # (B, NUM_JOINTS*3)

        pred_varen_params = {
            'global_orient': global_orient,
            'body_pose': body_pose,
            'betas': pred_betas,
        }
        return pred_varen_params, pred_cam, None
