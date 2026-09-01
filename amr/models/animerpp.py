import torch
import pytorch_lightning as pl
from torchvision.utils import make_grid
from typing import Dict
from yacs.config import CfgNode
from ..utils import MeshRenderer
from ..utils.geometry import perspective_projection, aa_to_rotmat
from ..utils.pylogger import get_pylogger
from .backbones import create_backbone
from .heads.classifier_head import ClassTokenHead
from .heads import build_varen_head
from .losses import (Keypoint3DLoss, Keypoint2DLoss, ParameterLoss, SupConLoss)
from .varen_warapper import VAREN


log = get_pylogger(__name__)


def _varen_native_to_camera_frame(x: torch.Tensor) -> torch.Tensor:
    """Put VAREN's raw output into the OpenCV camera frame the rest of this
    pipeline works in: +X right, +Y DOWN, +Z depth away from the camera."""
    return torch.stack([x[..., 0], -x[..., 2], x[..., 1]], dim=-1)


class AniMerPlusPlus(pl.LightningModule):
    def __init__(self, cfg: CfgNode, init_renderer: bool = True):
        """
        Setup VAREN horse model
        Args:
            cfg (CfgNode): Config file as a yacs CfgNode
        """
        super().__init__()

        # Save hyperparameters
        self.save_hyperparameters(logger=False, ignore=['init_renderer'])

        self.cfg = cfg
        # Create backbone feature extractor
        self.backbone = create_backbone(cfg)
        if cfg.MODEL.BACKBONE.get('PRETRAINED_WEIGHTS', None):
            weights_path = cfg.MODEL.BACKBONE.PRETRAINED_WEIGHTS
            log.info(f'Loading backbone weights from {weights_path}')
            checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
            if 'state_dict' in checkpoint:
                state_dict = {
                    k[len('backbone.'):]: v
                    for k, v in checkpoint['state_dict'].items()
                    if k.startswith('backbone.')
                }
            else:
                state_dict = {k.replace('backbone.', ''): v for k, v in checkpoint.items()}
            missing_keys, unexpected_keys = self.backbone.load_state_dict(state_dict, strict=False)
            log.info(f'Backbone weights loaded (missing_keys={len(missing_keys)}, unexpected_keys={len(unexpected_keys)})')
    
        # Create VAREN head
        self.varen_head = build_varen_head(cfg)

        self.class_token_head = ClassTokenHead(**cfg.MODEL.get("CLASS_TOKEN_HEAD", dict()))

        # Define loss functions
        self.keypoint_3d_loss = Keypoint3DLoss(loss_type='l1')
        self.keypoint_2d_loss = Keypoint2DLoss(loss_type='l1')
        self.supcon_loss = SupConLoss()
        self.parameter_loss = ParameterLoss()

        # Instantiate VAREN model
        varen_model_path = cfg.VAREN.MODEL_PATH
        self.varen = VAREN(model_path=varen_model_path,
                           num_betas=cfg.VAREN.get('NUM_BETAS', 39),
                           use_muscle_deformations=cfg.VAREN.get('USE_MUSCLE_DEFORMATIONS', False),
                           ext=cfg.VAREN.get('EXT', 'pkl'))

        # Buffer that shows whether we need to initialize ActNorm layers
        self.register_buffer('initialized', torch.tensor(False))
        # Setup renderer for visualization
        if init_renderer:
            import numpy as np
            self.varen_mesh_renderer = MeshRenderer(self.cfg, faces=np.asarray(self.varen.faces))
        else:
            self.renderer = None
            self.mesh_renderer = None

        self.automatic_optimization = False

    def get_parameters(self):
        all_params = list(self.varen_head.parameters())
        all_params += list(self.backbone.parameters())
        all_params += list(self.class_token_head.parameters())
        return all_params

    def configure_optimizers(self):
        """
        Setup model and distriminator Optimizers
        Returns:
            Tuple[torch.optim.Optimizer, torch.optim.Optimizer]: Model and discriminator optimizers
        """
        base_lr = self.cfg.TRAIN.LR
        lr_groups = self.cfg.TRAIN.get('BACKBONE_LR_GROUPS', None)

        if lr_groups and "vit" in self.cfg.MODEL.BACKBONE.TYPE:
            # discriminative LR across backbone depth -- earlier blocks keep
            # more of their pretrained features, later blocks adapt faster.
            # non-block params (patch_embed/pos_embed/cls_token) count as
            # block 0, same as _freeze_stages' own convention.
            import re
            block_re = re.compile(r'blocks\.(\d+)\.')
            by_mult: Dict[float, list] = {}
            for name, p in self.backbone.named_parameters():
                if not p.requires_grad:
                    continue
                m = block_re.match(name)
                block_idx = int(m.group(1)) if m else 0
                lr_mult = 1.0
                for group in lr_groups:
                    if block_idx <= group['max_block']:
                        lr_mult = group['lr_mult']
                        break
                by_mult.setdefault(lr_mult, []).append(p)
            param_groups = [{'params': params, 'lr': base_lr * lr_mult} for lr_mult, params in by_mult.items()]
            head_params = list(self.varen_head.parameters()) + list(self.class_token_head.parameters())
            param_groups.append({'params': filter(lambda p: p.requires_grad, head_params), 'lr': base_lr})
        else:
            param_groups = [{'params': filter(lambda p: p.requires_grad, self.get_parameters()), 'lr': base_lr}]

        if "vit" in self.cfg.MODEL.BACKBONE.TYPE:
            optimizer = torch.optim.AdamW(params=param_groups,
                                          weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)
        else:
            optimizer = torch.optim.Adam(params=param_groups,
                                         weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)

        # short linear warmup -> cosine decay over the full run (manual optimization,
        # so training_step steps this itself -- Lightning won't auto-step it)
        warmup_steps = self.cfg.TRAIN.get('WARMUP_STEPS', 500)
        total_steps = self.cfg.GENERAL.TOTAL_STEPS
        warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-3, total_iters=warmup_steps)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_steps - warmup_steps, 1))
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])
        return [optimizer], [scheduler]

    def forward_backbone(self, batch: Dict):
        x = batch['img']
        dataset_source = (batch.get("supercategory", None) < 5) if batch.get("supercategory", None) is not None else None
        # Compute conditioning features using the backbone
        if self.cfg.MODEL.BACKBONE.TYPE in ["vith"]:
            conditioning_feats, cls = self.backbone(x[:, :, :, 32:-32])  # [256, 192]
        elif self.cfg.MODEL.BACKBONE.TYPE in ["vithmoe"]:
            conditioning_feats, cls = self.backbone(x[:, :, :, 32:-32], dataset_source=dataset_source.type(torch.long))
        else:
            conditioning_feats = self.backbone(x)
            cls = None
        return conditioning_feats, cls

    def forward_one_parametric_model(self, 
                                     focal_length: torch.tensor, 
                                     features: torch.tensor,
                                     head: torch.nn.Module,
                                     parametric_model: torch.nn.Module,):
        """
        Run a forward step of one parametric model.
        Args:
            batch (Dict): Dictionary containing batch data
        Returns:
            Dict: Dictionary containing the regression output
        """
        batch_size = features.shape[0]
        pred_params, pred_cam, _ = head(features)
        # Store useful regression outputs to the output dict
        output = {}
        output['pred_cam'] = pred_cam
        output['pred_params'] = {k: v.clone() for k, v in pred_params.items()}

        # Compute camera translation
        cam_scale = torch.nn.functional.softplus(pred_cam[:, 0]) + 1e-3  # keep scale positive, avoid div-by-zero/sign-flip
        pred_cam_t = torch.stack([pred_cam[:, 1],
                                  pred_cam[:, 2],
                                  2 * focal_length[:, 0] / (self.cfg.MODEL.IMAGE_SIZE * cam_scale)], dim=-1)
        output['pred_cam_t'] = pred_cam_t
        output['focal_length'] = focal_length

        # Compute model vertices, joints and the projected joints.
        pred_params['betas'] = pred_params['betas'].reshape(batch_size, -1)
        if 'body_pose' in pred_params:
            betas = pred_params['betas']
            is_axis_angle = head.joint_rep_type == 'aa'
            if is_axis_angle:
                global_orient = pred_params['global_orient'].reshape(batch_size, 3)
                body_pose = pred_params['body_pose'].reshape(batch_size, -1)
            else:
                global_orient = pred_params['global_orient'].reshape(batch_size, 1, 3, 3)
                body_pose = pred_params['body_pose'].reshape(batch_size, -1, 3, 3)
            parametric_model_output = parametric_model(global_orient=global_orient,
                                                      body_pose=body_pose,
                                                      betas=betas,
                                                      transl=None,
                                                      pose2rot=is_axis_angle)
            # fix axis mismatch: put VAREN's raw output into the dataset's frame
            parametric_model_output.vertices = _varen_native_to_camera_frame(parametric_model_output.vertices)
            if getattr(parametric_model_output, 'surface_keypoints', None) is not None:
                parametric_model_output.surface_keypoints = _varen_native_to_camera_frame(parametric_model_output.surface_keypoints)
            if getattr(parametric_model_output, 'joints', None) is not None:
                parametric_model_output.joints = _varen_native_to_camera_frame(parametric_model_output.joints)
        else:
            pred_params['global_orient'] = pred_params['global_orient'].reshape(batch_size, -1, 3, 3)
            pred_params['pose'] = pred_params['pose'].reshape(batch_size, -1, 3, 3)
            pred_params['bone'] = pred_params['bone'].reshape(batch_size, -1) if 'bone' in pred_params else None
            parametric_model_output = parametric_model(**pred_params, pose2rot=False)

        surface_keypoints = getattr(parametric_model_output, 'surface_keypoints', None)
        pred_keypoints_3d = surface_keypoints if surface_keypoints is not None else parametric_model_output.joints
        pred_vertices = parametric_model_output.vertices
        output['pred_keypoints_3d'] = pred_keypoints_3d.reshape(batch_size, -1, 3)
        output['pred_vertices'] = pred_vertices.reshape(batch_size, -1, 3)
        pred_cam_t = pred_cam_t.reshape(-1, 3)
        focal_length = focal_length.reshape(-1, 2)
        pred_keypoints_2d = perspective_projection(pred_keypoints_3d,
                                                   translation=pred_cam_t,
                                                   focal_length=focal_length / self.cfg.MODEL.IMAGE_SIZE)
        output['pred_keypoints_2d'] = pred_keypoints_2d.reshape(batch_size, -1, 2)
        return output

    def forward_step(self, batch: Dict, train: bool = False) -> Dict:
        """
        Run a forward step of the network
        Args:
            batch (Dict): Dictionary containing batch data
            train (bool): Flag indicating whether it is training or validation mode
        Returns:
            Dict: Dictionary containing the regression output
        """
        # Use RGB image as input
        x = batch['img']
        batch_size = x.shape[0]
        device = x.device
        features, cls = self.forward_backbone(batch)

        output = dict()
        output['cls_feats'] = self.class_token_head(cls) if self.cfg.MODEL.BACKBONE.get("USE_CLS", False) else None

        output['varen_output'] = self.forward_one_parametric_model(batch['focal_length'],
                                                                   features,
                                                                   self.varen_head,
                                                                   self.varen)
        return output

    def compute_varen_loss(self, batch: Dict, output: Dict) -> torch.Tensor:
        """
        Compute VAREN losses given the input batch and the regression output.
        Args:
            batch (Dict): Dictionary containing batch data
            output (Dict): Dictionary containing the regression output
        Returns:
            torch.Tensor: Total loss for current batch and loss components.
        """
        pred_params = output['pred_params']
        pred_keypoints_2d = output['pred_keypoints_2d']
        pred_keypoints_3d = output['pred_keypoints_3d']
        batch_size = pred_keypoints_3d.shape[0]

        gt_keypoints_2d = batch['keypoints_2d']
        gt_keypoints_3d = batch['keypoints_3d']

        loss_keypoints_2d = self.keypoint_2d_loss(pred_keypoints_2d, gt_keypoints_2d)
        loss_keypoints_3d = self.keypoint_3d_loss(pred_keypoints_3d, gt_keypoints_3d, pelvis_id=0)

        # Explicit scale supervision: generated horses were coming out consistently
        # smaller than ground truth. Keypoint3DLoss's per-point L1 mixes pose and
        # size error together, so overall size gets a weak signal -- this isolates
        # it directly as mean pelvis-relative keypoint distance (pred vs GT).
        gt_conf = gt_keypoints_3d[:, :, -1]
        pred_rel = pred_keypoints_3d - pred_keypoints_3d[:, 0:1, :]
        gt_rel = gt_keypoints_3d[:, :, :-1] - gt_keypoints_3d[:, 0:1, :-1]
        pred_scale = (pred_rel.norm(dim=-1) * gt_conf).sum(dim=1) / gt_conf.sum(dim=1).clamp(min=1)
        gt_scale = (gt_rel.norm(dim=-1) * gt_conf).sum(dim=1) / gt_conf.sum(dim=1).clamp(min=1)
        loss_scale = (pred_scale - gt_scale).abs().sum()

        gt_params = batch['varen_params']
        has_params = batch['has_varen_params']
        is_axis_angle = batch['varen_params_is_axis_angle']
        loss_varen_params = {}
        for k, pred in pred_params.items():
            if k not in gt_params:
                continue
            gt = gt_params[k].view(batch_size, -1)
            if is_axis_angle.get(k, torch.zeros(1, dtype=torch.bool)).all() and pred.dim() == 4:
                gt = aa_to_rotmat(gt.reshape(-1, 3)).view(batch_size, -1, 3, 3)
            loss_varen_params[k] = self.parameter_loss(pred.reshape(batch_size, -1),
                                                       gt.reshape(batch_size, -1),
                                                       has_params[k])

        loss_config = self.cfg.LOSS_WEIGHTS.VAREN
        loss = loss_config['KEYPOINTS_3D'] * loss_keypoints_3d + \
               loss_config['KEYPOINTS_2D'] * loss_keypoints_2d + \
               loss_config['SCALE'] * loss_scale + \
               sum([loss_varen_params[k] * loss_config[k.upper()] for k in loss_varen_params])

        losses = dict(loss_varen=loss.detach(),
                      loss_varen_keypoints_2d=loss_keypoints_2d.detach(),
                      loss_varen_keypoints_3d=loss_keypoints_3d.detach(),
                      loss_varen_scale=loss_scale.detach())
        for k, v in loss_varen_params.items():
            losses['loss_varen_' + k] = v.detach()
        return loss, losses


    def compute_loss(self, batch: Dict, output: Dict, train: bool = True) -> torch.Tensor:
        """
        Compute losses given the input batch and the regression output
        Args:
            batch (Dict): Dictionary containing batch data
            output (Dict): Dictionary containing the regression output
            train (bool): Flag indicating whether it is training or validation mode
        Returns:
            torch.Tensor : Total loss for current batch
        """
        x = batch['img']
        device, dtype = x.device, x.dtype
        if 'varen_output' in output:
            loss_varen, losses_varen = self.compute_varen_loss(batch, output['varen_output'])
        else:
            loss_varen, losses_varen = torch.tensor(0.0, device=device, dtype=dtype), {}
        loss_supcon = self.supcon_loss(output['cls_feats'], labels=batch['category']) if self.cfg.MODEL.BACKBONE.get("USE_CLS", False) \
                      else torch.tensor(0.0, device=device, dtype=dtype)
        loss = loss_varen + loss_supcon * self.cfg.LOSS_WEIGHTS['SUPCON']

        # Saving loss
        losses = {}
        losses['loss'] = loss.detach()
        losses['loss_supcon'] = loss_supcon.detach()
        for k, v in losses_varen.items():
            losses[k] = v.detach()
        output['losses'] = losses
        return loss

    # Tensoroboard logging should run from first rank only
    @pl.utilities.rank_zero.rank_zero_only
    def tensorboard_logging(self, batch: Dict, output: Dict, step_count: int, train: bool = True,
                            write_to_summary_writer: bool = True) -> None:
        """
        Log results to Tensorboard
        Args:
            batch (Dict): Dictionary containing batch data
            output (Dict): Dictionary containing the regression output
            step_count (int): Global training step count
            train (bool): Flag indicating whether it is training or validation mode
        """

        mode = 'train' if train else 'val'
        batch_size = batch['keypoints_2d'].shape[0]
        images = batch['img']
        masks = batch['mask']
        # mul std then add mean
        images = (images) * (torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(1, 3, 1, 1))
        images = (images + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1, 3, 1, 1))
        masks = masks.unsqueeze(1).repeat(1, 3, 1, 1)

        gt_keypoints_2d = batch['keypoints_2d']
        losses = output['losses']
        if write_to_summary_writer:
            summary_writer = self.logger.experiment
            for loss_name, val in losses.items():
                summary_writer.add_scalar(mode + '/' + loss_name, val.detach().item(), step_count)
            if train is False:
                for metric_name, val in output['metric'].items():
                    summary_writer.add_scalar(mode + '/' + metric_name, val, step_count)
        
        rend_imgs = []
        num_images = min(batch_size, self.cfg.EXTRA.NUM_LOG_IMAGES)
        dataset_source = (batch["supercategory"] < 5)[:num_images]  # bird for index 0

        if 'varen_output' in output:
            rend_imgs_varen = self.varen_mesh_renderer.visualize_tensorboard(
                                                                            output['varen_output']['pred_vertices'].detach().float().cpu().numpy()[:num_images],
                                                                            output['varen_output']['pred_cam_t'].detach().float().cpu().numpy()[:num_images],
                                                                            images[:num_images].float().cpu().numpy(),
                                                                            self.cfg.VAREN.get("FOCAL_LENGTH", 1000),
                                                                            output['varen_output']['pred_keypoints_2d'].detach().float().cpu().numpy()[:num_images],
                                                                            gt_keypoints_2d[:num_images].float().cpu().numpy(),
                                                                            )
            rend_imgs.extend(rend_imgs_varen)

        rend_imgs = make_grid(rend_imgs, nrow=5, padding=2)
        if write_to_summary_writer:
            summary_writer.add_image('%s/predictions' % mode, rend_imgs, step_count)

        return rend_imgs

    def forward(self, batch: Dict) -> Dict:
        """
        Run a forward step of the network in val mode
        Args:
            batch (Dict): Dictionary containing batch data
        Returns:
            Dict: Dictionary containing the regression output
        """
        return self.forward_step(batch, train=False)

    def training_step(self, batch: Dict) -> Dict:
        """
        Run a full training step
        Args:
            batch (Dict): Dictionary containing batch tensors such as 'img', 'mask', 'keypoints_2d', 'keypoints_3d', 'category', 'supercategory', and 'focal_length'.
        Returns:
            Dict: Dictionary containing regression output.
        """
        batch = batch['img']
        optimizer = self.optimizers(use_pl_optimizer=True)

        batch_size = batch['img'].shape[0]
        output = self.forward_step(batch, train=True)
        if self.cfg.get('UPDATE_GT_SPIN', False):
            self.update_batch_gt_spin(batch, output)
        loss = self.compute_loss(batch, output, train=True)

        # Error if Nan
        if torch.isnan(loss):
            raise ValueError('Loss is NaN')

        optimizer.zero_grad()
        self.manual_backward(loss)
        # Clip gradient
        if self.cfg.TRAIN.get('GRAD_CLIP_VAL', 0) > 0:
            gn = torch.nn.utils.clip_grad_norm_(self.get_parameters(), self.cfg.TRAIN.GRAD_CLIP_VAL,
                                                error_if_nonfinite=True)
            self.log('train/grad_norm', gn, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        optimizer.step()
        self.lr_schedulers().step()
        if self.global_step > 0 and self.global_step % self.cfg.GENERAL.LOG_STEPS == 0:
            self.tensorboard_logging(batch, output, self.global_step, train=True)

        self.log('train/loss', output['losses']['loss'], on_step=True, on_epoch=True, prog_bar=True, logger=False,
                 batch_size=batch_size, sync_dist=True)

        return output

    def validation_step(self, batch: Dict, batch_idx: int, dataloader_idx=0) -> Dict:
        pass
