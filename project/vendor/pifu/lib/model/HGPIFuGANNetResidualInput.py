import torch
import torch.nn as nn
from pdb import set_trace as st

from .HGPIFuGANNet import HGPIFuNetGAN
from project.models.stylesdf_model import EqualLinear
from project.models.helper_modules.helpers import conv3x3, ResidualBlock, conv1x1, conv
from project.models.helper_modules.resnetfc import ResnetBlockFC


def constant_init(module, val, bias=0):
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


# * for 3D GAN-based pixel-aligned & global-prior driven training
class HGPIFuNetGANResidual(HGPIFuNetGAN):

    def __init__(self,
                 opt,
                 opt_stylesdf,
                 projection_mode='orthogonal',
                 error_term=nn.L1Loss()):
        super().__init__(opt, opt_stylesdf, projection_mode, error_term)

        self.downsample_channel_conv = conv(
            512, 64, 3, 1, norm='bn')  # channel downsample img feature maps

        depth_dim = 32

        # norm_layer = lambda dim: nn.InstanceNorm2d(
        #     dim, track_running_stats=False, affine=True)

        # TODO change back to BN
        self.depth_conv = nn.Sequential(
            conv3x3(1, depth_dim),
            # ResidualBlock(depth_dim, depth_dim, norm_layer=norm_layer),
            ResidualBlock(depth_dim, depth_dim),
            conv1x1(depth_dim, depth_dim),
        )
        self.residual_conv = nn.Sequential(
            conv3x3(3, depth_dim),
            # ResidualBlock(depth_dim, depth_dim, norm_layer=norm_layer),
            ResidualBlock(depth_dim, depth_dim),
            conv1x1(depth_dim, depth_dim),
        )

    def filter(self,
               residual_images,
               depth_feat=None,
               ref_feats=None,
               feat_key='ref_view',
               *args,
               **kwargs):
        '''
        Filter the input images
        store all intermediate features.
        :param images: [B, C, H, W] input images
        '''

        feats_to_filter = self.residual_conv(residual_images)  # 3 -> 32

        if ref_feats is not None:
            feats_to_filter = torch.cat(
                (feats_to_filter, self.downsample_channel_conv(ref_feats)),
                dim=1)  # 512 -> 64
            raise DeprecationWarning('deprecated in editing version model.')

        if depth_feat is not None:
            feats_to_filter = torch.cat(
                (feats_to_filter, self.depth_conv(depth_feat)),
                1)  # 128 dim input

        return super().filter(feats_to_filter,
                              feat_key=feat_key,
                              *args,
                              **kwargs)

    def build_modulation_net(self, opt_stylesdf):

        if opt_stylesdf.L_pred_geo_modulations:
            self.local_feat_to_geo_modulations_linear = EqualLinear(
                256, 256 * 2)  # with depth

        if opt_stylesdf.L_pred_tex_modulations:
            # TODO, just test whether fusion bug
            self.local_feat_to_tex_modulations_linear = EqualLinear(
                opt_stylesdf.residual_local_feats_dim, 256 * 2)  # with depth
            constant_init(self.local_feat_to_tex_modulations_linear,
                          val=0,
                          bias=0)

        # if opt_stylesdf.L_pred_tex_modulations:
        #     self.local_feat_to_tex_modulations_linear = ResnetBlockFC(
        #         opt_stylesdf.residual_local_feats_dim, 256 * 2)

        #     nn.init.zeros_(self.local_feat_to_tex_modulations_linear.fc_0.bias)
        #     nn.init.zeros_(self.local_feat_to_tex_modulations_linear.fc_0.weight)
        #     nn.init.zeros_(self.local_feat_to_tex_modulations_linear.fc_1.bias)
        #     if self.local_feat_to_tex_modulations_linear.shortcut is not None:
        #         nn.init.zeros_(self.local_feat_to_tex_modulations_linear.shortcut.weight)
