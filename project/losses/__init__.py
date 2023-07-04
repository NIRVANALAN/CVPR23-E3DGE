# 2d reconstruction losses
# gan losses
from .gan_loss import (d_logistic_loss, d_r1_loss, eikonal_loss,
                       g_nonsaturating_loss, g_path_regularize,
                       viewpoints_loss)
# 3d metrics
# from .geometry_loss import chamfer_distance
from .id_loss import IDLoss
from .lpips.lpips import LPIPS

# manage import
__all__ = [
    'LPIPS',
    'IDLoss',
    'viewpoints_loss',
    'eikonal_loss',
    'd_logistic_loss',
    'd_r1_loss',
    'g_nonsaturating_loss',
    'g_path_regularize',
]
