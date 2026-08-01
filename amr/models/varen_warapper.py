import torch
from typing import Optional, NewType
from varen import VAREN as VARENModel

Tensor = NewType('Tensor', torch.Tensor)


class VAREN(torch.nn.Module):
    def __init__(self,
                 model_path: str,
                 num_betas: int = 39,
                 use_muscle_deformations: bool = False,
                 ext: str = 'pkl',
                 **kwargs):
        super().__init__()
        self.model = VARENModel(model_path,
                                num_betas=num_betas,
                                use_muscle_deformations=use_muscle_deformations,
                                ext=ext,
                                **kwargs)
        self.faces = self.model.faces

    def forward(self,
                global_orient: Tensor,
                body_pose: Tensor,
                betas: Tensor,
                transl: Optional[Tensor] = None,
                pose2rot: bool = True,
                **kwargs):
        return self.model(betas=betas,
                          body_pose=body_pose,
                          global_orient=global_orient,
                          transl=transl,
                          pose2rot=pose2rot)
