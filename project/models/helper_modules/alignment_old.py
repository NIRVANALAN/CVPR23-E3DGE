import functools
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn
from torch.nn import BatchNorm2d, Conv2d, Module, PReLU, Sequential

from project.models.helper_modules import helpers
from .helpers import bottleneck_IR, DemodulatedConv2d, ResidualBlock, conv, conv1x1, conv3x3
# from .project.models.stylesdf_model import EqualConv2d, ScaledLeakyReLU
from project.models.stylesdf_model import EqualConv2d, ScaledLeakyReLU


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d,
                                       affine=True,
                                       track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d,
                                       affine=False,
                                       track_running_stats=False)
    elif norm_type == 'none':

        def norm_layer(x):
            return nn.Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' %
                                  norm_type)
    return norm_layer


class CorrelationVolume(nn.Module):
    """
    Implementation by Ignacio Rocco
    paper: https://arxiv.org/abs/1703.05593
    project: https://github.com/ignacio-rocco/cnngeometric_pytorch
    """

    def __init__(self):
        super(CorrelationVolume, self).__init__()

    def forward(self, feature_A, feature_B):
        b, c, h, w = feature_A.size()

        # reshape features for matrix multiplication
        feature_A = feature_A.transpose(2, 3).contiguous().reshape(b, c, h * w)
        feature_B = feature_B.reshape(b, c, h * w).transpose(1, 2)
        feature_mul = torch.bmm(feature_B, feature_A)
        correlation_tensor = feature_mul.reshape(b, h, w, h * w).transpose(
            2, 3).transpose(1, 2)
        return correlation_tensor


class FeatureL2Norm(nn.Module):
    """
    Implementation by Ignacio Rocco
    paper: https://arxiv.org/abs/1703.05593
    project: https://github.com/ignacio-rocco/cnngeometric_pytorch
    """

    def __init__(self):
        super(FeatureL2Norm, self).__init__()

    def forward(self, feature):
        epsilon = 1e-6
        norm = torch.pow(torch.sum(torch.pow(feature, 2), 1) + epsilon,
                         0.5).unsqueeze(1).expand_as(feature)
        return torch.div(feature, norm)


def conv_blck(in_channels,
              out_channels,
              kernel_size=3,
              stride=1,
              padding=1,
              dilation=1,
              bn=False):
    if bn:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                      dilation), nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
    else:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                      dilation), nn.ReLU(inplace=True))


def conv_head(in_channels):
    return nn.Conv2d(in_channels, 2, kernel_size=3, padding=1)


class CorrespondenceMapBase(nn.Module):

    def __init__(self, in_channels, bn=False):
        super().__init__()

    def forward(self, x1, x2=None, x3=None):
        x = x1
        # concatenating dimensions
        if (x2 is not None) and (x3 is None):
            x = torch.cat((x1, x2), 1)
        elif (x2 is None) and (x3 is not None):
            x = torch.cat((x1, x3), 1)
        elif (x2 is not None) and (x3 is not None):
            x = torch.cat((x1, x2, x3), 1)

        return x


class CMD60x60(CorrespondenceMapBase):

    def __init__(self, in_channels, bn=False):
        super().__init__(in_channels, bn)
        # number of output channels
        chan = [128, 96, 64, 32]
        self.conv0 = conv_blck(in_channels, chan[0], bn=bn)  # ? how to tune
        self.conv1 = conv_blck(chan[0], chan[1], padding=2, dilation=2, bn=bn)
        self.conv2 = conv_blck(chan[1], chan[2], padding=3, dilation=3, bn=bn)
        self.conv3 = conv_blck(chan[2], chan[3], padding=4, dilation=4, bn=bn)
        self.final = conv_head(chan[-1])

    def forward(self, x1, x2=None, x3=None):
        x = super().forward(x1, x2, x3)
        x = self.conv3(self.conv2(self.conv1(self.conv0(x))))
        return self.final(x)


# ! todo
def factory(in_channels):
    # if type == 'level_0':
    #     return CMD240x240(in_channels, True)
    # elif type == 'level_1':
    #     return CMD120x120(in_channels=in_channels, bn=True)
    # elif type == 'level_2':
    #     return CMD60x60(in_channels=in_channels, bn=True)
    # elif type == 'level_3':
    #     return CMDTop(in_channels=in_channels, bn=True)
    # elif type == 'level_4':
    #     return CMDTop(in_channels=in_channels, bn=True)
    # assert 0, 'Correspondence Map Decoder bad creation: ' + type
    return CMD60x60(in_channels=in_channels,
                    bn=True)  # H W = 64, just use this one here.


class GANBasedDGCNet(Module):
    # gan based dense geometry correspondence network, DGC-Net like architecture
    def __init__(self, device) -> None:
        super().__init__()
        self.device = device
        self.mask = False

        # self.pyramid = VGGPyramid().to(self.device)
        # L2 feature normalisation
        self.l2norm = FeatureL2Norm().to(self.device)
        # Correlation volume
        self.corr = CorrelationVolume().to(self.device)

        # create a hierarchy of correspondence decoders
        map_dim = 2
        # N_out = [x + map_dim for x in [128, 128, 256, 512, 225]]
        N_out = [x + map_dim for x in [64**2, 512, 512, 512]]  # 1 3 5 7.
        self.pyramid_level = len(N_out)
        # todo, need to reduce channels.

        for i, in_chan in enumerate(N_out):
            self.__dict__['_modules']['reg_' +
                                      str(i)] = factory(in_chan).to(device)

    # def forward(self, x1, x2):
    def forward(self, target_pyr, source_pyr):
        """
        x1 - target image
        x2 - source image
        """

        # target_pyr = self.pyramid(x1)
        # source_pyr = self.pyramid(x2)
        # ! pyr here, shallow -> higher. reverse compared with vgg.

        # do feature normalisation
        feat_top_pyr_trg = self.l2norm(target_pyr[0])  # B C H W
        feat_top_pyr_src = self.l2norm(source_pyr[0])

        # do correlation
        corr1 = self.corr(feat_top_pyr_trg, feat_top_pyr_src)  # B H*W H W
        corr1 = self.l2norm(F.relu(corr1))

        b, c, h, w = corr1.size()
        init_map = torch.FloatTensor(b, 2, h, w).zero_().to(self.device)
        est_grid = self.__dict__['_modules'][
            'reg_0'](  # 0 = original coarse layer here
                x1=corr1, x3=init_map)

        estimates_grid = [est_grid]
        '''
        create correspondence map decoder, upsampler for each level of
        the feature pyramid
        '''
        # for k in reversed(range(self.pyramid_level - 1)):
        for k in range(1, self.pyramid_level):
            p1, p2 = target_pyr[k], source_pyr[k]
            # est_map = F.interpolate(input=estimates_grid[-1], scale_factor=2, mode='bilinear', align_corners=False)
            est_map = estimates_grid[-1]

            p1_w = F.grid_sample(p1, est_map.transpose(1, 2).transpose(2, 3))
            est_map = self.__dict__['_modules']['reg_' + str(k)](x1=p1_w,
                                                                 x2=p2,
                                                                 x3=est_map)
            estimates_grid.append(est_map)

        return estimates_grid


class GANBasedDGCNetV2(Module):
    # gan based dense geometry correspondence network, DGC-Net like architecture
    def __init__(self, device) -> None:
        pass


# class CostVolumeInitNet(nn.Module):
class AlignInpainter(nn.Module):

    def __init__(self, ):
        super().__init__()

        norm_layer = lambda dim: nn.InstanceNorm2d(
            dim, track_running_stats=False, affine=True)

        featmap_dim = 256  # warped_feats channel
        edit_img_dim = 32  # img target channel

        self.edit_img_conv = nn.Sequential(
            conv3x3(3, edit_img_dim),
            ResidualBlock(edit_img_dim, edit_img_dim, norm_layer=norm_layer),
            conv1x1(edit_img_dim, edit_img_dim),
        )
        # featmap_dim += edit_img_dim

        self.out_conv = nn.Sequential(
            # conv3x3(featmap_dim, featmap_dim),
            ResidualBlock(featmap_dim + edit_img_dim,
                          featmap_dim,
                          norm_layer=norm_layer),
            conv1x1(featmap_dim, featmap_dim),  # why 
        )

    def forward(self, warped_feats, edit_img):
        warped_feats = warped_feats.squeeze(-2).permute(
            0, 3, 1, 2)  # B H W 1 C -> B H W C
        edit_img_feats = self.edit_img_conv(edit_img)

        inpainted_feats = self.out_conv(
            torch.cat([warped_feats, edit_img_feats], 1))
        feats_out = warped_feats + inpainted_feats  # B C H W
        feats_out = feats_out.permute(0, 2, 3, 1).unsqueeze(-2)
        return feats_out


class AlignInpainterLite(nn.Module):

    def __init__(self, ):
        super().__init__()

        norm_layer = lambda dim: nn.InstanceNorm2d(
            dim, track_running_stats=False, affine=True)

        featmap_dim = 256  # warped_feats channel
        edit_img_dim = 32  # img target channel

        self.edit_img_conv = nn.Sequential(
            conv3x3(3, edit_img_dim),
            ResidualBlock(edit_img_dim, edit_img_dim, norm_layer=norm_layer),
            conv1x1(edit_img_dim, edit_img_dim),
        )
        # featmap_dim += edit_img_dim

        self.out_conv = nn.Sequential(
            # conv3x3(featmap_dim, featmap_dim),
            ResidualBlock(featmap_dim + edit_img_dim,
                          featmap_dim,
                          norm_layer=norm_layer),
            conv1x1(featmap_dim, featmap_dim),  # why 
        )

    def forward(self, proj_feats, edit_img, reshape=True):
        assert proj_feats.ndim == 4
        # if proj_feats.ndim==5:
        #     proj_feats = proj_feats.squeeze(-2)

        if proj_feats.shape[-2] != proj_feats.shape[-1]:  # B C H W
            proj_feats = proj_feats.permute(0, 3, 1, 2)  # B H W C -> B C H W

        edit_img_feats = self.edit_img_conv(edit_img)

        inpainted_proj_feats = self.out_conv(
            torch.cat([proj_feats, edit_img_feats], 1))
        if reshape:
            inpainted_proj_feats = inpainted_proj_feats.permute(
                0, 2, 3, 1).unsqueeze(-2)  # B H W 1 C
        return inpainted_proj_feats


# ADA, copied from HFGI https://github.dev/Tengfei-Wang/HFGI
class ResidualAligner(Module):

    def __init__(self, opt_training=None):
        super(ResidualAligner, self).__init__()
        if opt_training.aligner_demodulate:
            norm_layer = get_norm_layer('none')
        else:
            norm_layer = get_norm_layer(opt_training.aligner_norm_type)

        bottleneck_IR = partial(helpers.bottleneck_IR,
                                norm_layer=norm_layer,
                                demodulate=opt_training.aligner_demodulate)
        # if opt_training.remove_bn:
        print('norm_layer in ResidualAligner: {}'.format(
            opt_training.aligner_norm_type))

        if opt_training.aligner_demodulate:
            self.conv_layer1 = Sequential(
                DemodulatedConv2d(6, 16, (3, 3), 1, 1, bias=False),
                #   BatchNorm2d(16),
                norm_layer(16),
                PReLU(16))
        else:
            self.conv_layer1 = Sequential(
                Conv2d(6, 16, (3, 3), 1, 1, bias=False),
                #   BatchNorm2d(16),
                norm_layer(16),
                PReLU(16))

        self.conv_layer2 = Sequential(*[
            bottleneck_IR(16, 32, 2),
            bottleneck_IR(32, 32, 1),
            bottleneck_IR(32, 32, 1)
        ])
        self.conv_layer3 = Sequential(*[
            bottleneck_IR(32, 48, 2),
            bottleneck_IR(48, 48, 1),
            bottleneck_IR(48, 48, 1)
        ])
        self.conv_layer4 = Sequential(*[
            bottleneck_IR(48, 64, 2),
            bottleneck_IR(64, 64, 1),
            bottleneck_IR(64, 64, 1)
        ])

        self.dconv_layer1 = Sequential(*[
            bottleneck_IR(112, 64, 1),
            bottleneck_IR(64, 32, 1),
            bottleneck_IR(32, 32, 1)
        ])
        self.dconv_layer2 = Sequential(*[
            bottleneck_IR(64, 32, 1),
            bottleneck_IR(32, 16, 1),
            bottleneck_IR(16, 16, 1)
        ])
        self.dconv_layer3 = Sequential(*[
            bottleneck_IR(32, 16, 1),
            bottleneck_IR(16, 3, 1),
            bottleneck_IR(3, 3, 1)
        ])

    def forward(self, x):
        feat1 = self.conv_layer1(x)
        feat2 = self.conv_layer2(feat1)
        feat3 = self.conv_layer3(feat2)
        feat4 = self.conv_layer4(feat3)

        feat4 = torch.nn.functional.interpolate(feat4,
                                                size=(64, 64),
                                                mode='bilinear')
        dfea1 = self.dconv_layer1(torch.cat((feat4, feat3), 1))
        dfea1 = torch.nn.functional.interpolate(dfea1,
                                                size=(128, 128),
                                                mode='bilinear')
        dfea2 = self.dconv_layer2(torch.cat((dfea1, feat2), 1))
        dfea2 = torch.nn.functional.interpolate(dfea2,
                                                size=(256, 256),
                                                mode='bilinear')
        dfea3 = self.dconv_layer3(torch.cat((dfea2, feat1), 1))

        res_aligned = dfea3

        return res_aligned


# Consultation encoder
class ResidualEncoder(Module):

    def __init__(self, opts=None):
        super(ResidualEncoder, self).__init__()
        self.conv_layer1 = Sequential(Conv2d(3, 32, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(32), PReLU(32))

        self.conv_layer2 = Sequential(*[
            bottleneck_IR(32, 48, 2),
            bottleneck_IR(48, 48, 1),
            bottleneck_IR(48, 48, 1)
        ])

        self.conv_layer3 = Sequential(*[
            bottleneck_IR(48, 64, 2),
            bottleneck_IR(64, 64, 1),
            bottleneck_IR(64, 64, 1)
        ])

        self.condition_scale3 = nn.Sequential(
            EqualConv2d(64, 512, 3, stride=1, padding=1, bias=True),
            ScaledLeakyReLU(0.2),
            EqualConv2d(512, 512, 3, stride=1, padding=1, bias=True))

        self.condition_shift3 = nn.Sequential(
            EqualConv2d(64, 512, 3, stride=1, padding=1, bias=True),
            ScaledLeakyReLU(0.2),
            EqualConv2d(512, 512, 3, stride=1, padding=1, bias=True))

    def get_deltas_starting_dimensions(self):
        ''' Get a list of the initial dimension of every delta from which it is applied '''
        return list(range(
            self.style_count))  # Each dimension has a delta applied to it

    def forward(self, x):
        conditions = []
        feat1 = self.conv_layer1(x)
        feat2 = self.conv_layer2(feat1)
        feat3 = self.conv_layer3(feat2)

        scale = self.condition_scale3(feat3)
        scale = torch.nn.functional.interpolate(scale,
                                                size=(64, 64),
                                                mode='bilinear')
        conditions.append(scale.clone())
        shift = self.condition_shift3(feat3)
        shift = torch.nn.functional.interpolate(shift,
                                                size=(64, 64),
                                                mode='bilinear')
        conditions.append(shift.clone())
        return conditions


class AlignInpainterFusionBlock(nn.Module):

    def __init__(self):
        super().__init__()

        norm_layer = lambda dim: nn.InstanceNorm2d(
            dim, track_running_stats=False, affine=True)

        featmap_dim = 256  # warped_feats channel

        # if add_input_conv:
        # self.fusionBlock = nn.Sequential(
        #     conv3x3(featmap_dim, featmap_dim),
        #     ResidualBlock(featmap_dim, featmap_dim, norm_layer=norm_layer),
        #     # conv1x1(featmap_dim, featmap_dim), # has an MLP appended, no need here.
        # )
        # else:
        self.fusionBlock = nn.Sequential(
            # conv3x3(featmap_dim, featmap_dim),
            ResidualBlock(featmap_dim, featmap_dim, norm_layer=norm_layer),
            # conv1x1(featmap_dim, featmap_dim), # has an MLP appended, no need here.
        )

    def forward(self,
                proj_3dfeats,
                inpainted_2dfeats,
                visibility_mask,
                reshape=True):
        """proj_3dfeats: queried feats from reference view.
        inpaitned_2dfeats: 2D inpainting features, repeated.
        motivation: maintain original proj_3dfeats as much as possible & fusion within the same "modality"
        """
        assert proj_3dfeats.ndim == 5

        fused_feats = visibility_mask * proj_3dfeats + (
            1 - visibility_mask) * inpainted_2dfeats
        fused_feats = self.fusionBlock(fused_feats)

        if reshape:
            fused_feats = fused_feats.permute(0, 2, 3,
                                              1).unsqueeze(-2)  # B H W 1 C
        return fused_feats


# align feats, res_gt, que_depth, que_thumb_img
class FeatureAligner(Module):

    def __init__(self, opts=None, input_dim=32):
        super().__init__()

        # 16 + 3 * 8 = 40
        self.downsample_channel_conv = conv(
            512, 8, 3, 1)  # channel downsample img feature maps

        depth_dim = 8
        norm_layer = lambda dim: nn.InstanceNorm2d(
            dim, track_running_stats=False, affine=True)

        self.residual_conv = nn.Sequential(
            conv3x3(3, depth_dim),
            ResidualBlock(depth_dim, depth_dim, norm_layer=norm_layer),
            conv1x1(depth_dim, depth_dim),
        )

        self.depth_conv = nn.Sequential(
            conv3x3(1, depth_dim),
            ResidualBlock(depth_dim, depth_dim, norm_layer=norm_layer),
            conv1x1(depth_dim, depth_dim),
        )

        self.que_thumb_conv = nn.Sequential(
            conv3x3(3, depth_dim),
            ResidualBlock(depth_dim, depth_dim, norm_layer=norm_layer),
            conv1x1(depth_dim, depth_dim),
        )

        # self.conv_layer1 = Sequential(Conv2d(32, 32, (3, 3), 1, 1, bias=False),
        self.conv_layer1 = Sequential(
            Conv2d(input_dim, 32, (3, 3), 1, 1, bias=False), BatchNorm2d(32),
            PReLU(32))

        # self.conv_layer2 = Sequential(*[
        #     bottleneck_IR(16, 32, 2),
        #     bottleneck_IR(32, 32, 1),
        #     bottleneck_IR(32, 32, 1)
        # ])
        self.conv_layer2 = Sequential(*[
            bottleneck_IR(32, 48, 2),
            bottleneck_IR(48, 48, 1),
            bottleneck_IR(48, 48, 1)
        ])
        self.conv_layer3 = Sequential(*[
            bottleneck_IR(48, 64, 2),
            bottleneck_IR(64, 64, 1),
            bottleneck_IR(64, 64, 1)
        ])

        self.dconv_layer1 = Sequential(*[
            bottleneck_IR(112, 64, 1),
            bottleneck_IR(64, 32, 1),
            bottleneck_IR(32, 32, 1)
        ])
        self.dconv_layer2 = Sequential(*[
            bottleneck_IR(64, 32, 1),
            bottleneck_IR(32, 32, 1),
            # bottleneck_IR(16, 16, 1)
        ])
        # self.dconv_layer3 = Sequential(*[
        #     bottleneck_IR(32, 16, 1),
        #     bottleneck_IR(16, 3, 1),
        #     bottleneck_IR(3, 3, 1)
        # ])

    def forward(self, residual_images, que_depth, ref_feats, que_thumb_images):

        feats_to_filter = self.residual_conv(residual_images)  # 3 -> 32

        if ref_feats is not None:
            feats_to_filter = torch.cat(
                (feats_to_filter, self.downsample_channel_conv(ref_feats)),
                dim=1)  # 512 -> 64

        if que_depth is not None:
            feats_to_filter = torch.cat(
                (feats_to_filter, self.depth_conv(que_depth)),
                1)  # 128 dim input

        if que_thumb_images is not None:
            feats_to_filter = torch.cat(
                (feats_to_filter, self.que_thumb_conv(que_thumb_images)),
                1)  # 128 dim input

        feat1 = self.conv_layer1(feats_to_filter)  # 32
        # feat2 = self.conv_layer2(feat1)
        feat2 = self.conv_layer2(feat1)  # 48
        feat3 = self.conv_layer3(feat2)  # 64
        # feat4 = self.conv_layer4(feat1)

        feat3 = torch.nn.functional.interpolate(feat3,
                                                size=(128, 128),
                                                mode='bilinear')
        dfea1 = self.dconv_layer1(torch.cat((feat3, feat2), 1))  # 112 -> 32
        dfea1 = torch.nn.functional.interpolate(dfea1,
                                                size=(256, 256),
                                                mode='bilinear')
        dfea2 = self.dconv_layer2(torch.cat((dfea1, feat1),
                                            1))  # 32 + 32 -> 16
        # dfea2 = torch.nn.functional.interpolate(dfea2,
        #                                         size=(256, 256),
        #                                         mode='bilinear')
        # dfea3 = self.dconv_layer3(torch.cat((dfea2, feat1), 1))

        res_aligned = dfea2

        return res_aligned


# align feats, res_gt, que_depth, que_thumb_img
class FeatureAlignerBig(Module):

    def __init__(self, opts=None, input_dim=48):
        super().__init__()

        # 24 + 24 = 48
        self.downsample_channel_conv = conv(
            512, 24, 3, 1)  # channel downsample img feature maps

        depth_dim = 8
        norm_layer = lambda dim: nn.InstanceNorm2d(
            dim, track_running_stats=False, affine=True)

        self.residual_conv = nn.Sequential(
            conv3x3(3, depth_dim),
            ResidualBlock(depth_dim, depth_dim, norm_layer=norm_layer),
            conv1x1(depth_dim, depth_dim),
        )

        self.depth_conv = nn.Sequential(
            conv3x3(1, depth_dim),
            ResidualBlock(depth_dim, depth_dim, norm_layer=norm_layer),
            conv1x1(depth_dim, depth_dim),
        )

        self.que_thumb_conv = nn.Sequential(
            conv3x3(3, depth_dim),
            ResidualBlock(depth_dim, depth_dim, norm_layer=norm_layer),
            conv1x1(depth_dim, depth_dim),
        )

        # self.conv_layer1 = Sequential(Conv2d(32, 32, (3, 3), 1, 1, bias=False),
        self.conv_layer1 = Sequential(
            Conv2d(input_dim, 64, (3, 3), 1, 1, bias=False), BatchNorm2d(64),
            PReLU(64))

        # self.conv_layer2 = Sequential(*[
        #     bottleneck_IR(16, 32, 2),
        #     bottleneck_IR(32, 32, 1),
        #     bottleneck_IR(32, 32, 1)
        # ])
        self.conv_layer2 = Sequential(*[
            bottleneck_IR(64, 80, 2),
            bottleneck_IR(80, 80, 1),
            bottleneck_IR(80, 80, 1)
        ])
        self.conv_layer3 = Sequential(*[
            bottleneck_IR(80, 112, 2),
            bottleneck_IR(112, 112, 1),
            bottleneck_IR(112, 112, 1)
        ])

        self.dconv_layer1 = Sequential(*[
            bottleneck_IR(192, 112, 1),
            bottleneck_IR(112, 64, 1),
            bottleneck_IR(64, 64, 1)
        ])
        self.dconv_layer2 = Sequential(*[
            bottleneck_IR(64 + 64, 64, 1),
            bottleneck_IR(64, 64, 1),
            # bottleneck_IR(16, 16, 1)
        ])
        # self.dconv_layer3 = Sequential(*[
        #     bottleneck_IR(32, 16, 1),
        #     bottleneck_IR(16, 3, 1),
        #     bottleneck_IR(3, 3, 1)
        # ])

    def forward(self, residual_images, que_depth, ref_feats, que_thumb_images):

        feats_to_filter = self.residual_conv(residual_images)  # 3 -> 32

        if ref_feats is not None:
            feats_to_filter = torch.cat(
                (feats_to_filter, self.downsample_channel_conv(ref_feats)),
                dim=1)  # 512 -> 64

        if que_depth is not None:
            feats_to_filter = torch.cat(
                (feats_to_filter, self.depth_conv(que_depth)),
                1)  # 128 dim input

        if que_thumb_images is not None:
            feats_to_filter = torch.cat(
                (feats_to_filter, self.que_thumb_conv(que_thumb_images)),
                1)  # 128 dim input

        feat1 = self.conv_layer1(feats_to_filter)  # 64
        # feat2 = self.conv_layer2(feat1)
        feat2 = self.conv_layer2(feat1)  # 80
        feat3 = self.conv_layer3(feat2)  # 112
        # feat4 = self.conv_layer4(feat1)

        feat3 = torch.nn.functional.interpolate(feat3,
                                                size=(128, 128),
                                                mode='bilinear')
        dfea1 = self.dconv_layer1(torch.cat((feat3, feat2), 1))  # 192 -> 64
        dfea1 = torch.nn.functional.interpolate(dfea1,
                                                size=(256, 256),
                                                mode='bilinear')
        dfea2 = self.dconv_layer2(torch.cat((dfea1, feat1),
                                            1))  # 64 + 64 -> 64
        # dfea2 = torch.nn.functional.interpolate(dfea2,
        #                                         size=(256, 256),
        #                                         mode='bilinear')
        # dfea3 = self.dconv_layer3(torch.cat((dfea2, feat1), 1))

        res_aligned = dfea2

        return res_aligned
