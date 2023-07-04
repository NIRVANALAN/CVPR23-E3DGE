import math
from functools import partial
from pdb import set_trace as st

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn
from torch.nn import BatchNorm2d, Conv2d, Module, PReLU, Sequential

from .. import helpers
from ..helpers import DemodulatedConv2d, bottleneck_IR
from .pix2pix_networks import get_norm_layer

# from .stylegan2.model import EqualConv2d, EqualLinear, ScaledLeakyReLU

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
                    #   dilation), nn.BatchNorm2d(out_channels, track_running_stats=False),
                    #   dilation), nn.BatchNorm2d(out_channels, track_running_stats=False),
            nn.ReLU(inplace=True))
    else:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                      dilation), nn.ReLU(inplace=True))


def conv_head(in_channels):
    return nn.Conv2d(in_channels, 2, kernel_size=3, padding=1)

# ADA, copied from HFGI https://github.dev/Tengfei-Wang/HFGI
class ResidualAligner(Module):
    def __init__(self, norm_type='batch'):
        super(ResidualAligner, self).__init__()
        # if opt_train.aligner_demodulate:
        #     norm_layer = get_norm_layer('none')
        # else:
        norm_layer = get_norm_layer(norm_type)

        bottleneck_IR = partial(helpers.bottleneck_IR, norm_layer=norm_layer, demodulate=norm_type == 'none')
        # if opt_train.remove_bn:
        print('norm_layer in ResidualAligner: {}'.format(norm_type))

        # if norm_type == 'none':
        #     self.conv_layer1 = Sequential(DemodulatedConv2d(6, 16, (3, 3), 1, 1, bias=False),
        #                                 #   BatchNorm2d(16), 
        #                                 norm_layer(16),
        #                                 PReLU(16))
        # else:
        self.conv_layer1 = Sequential(Conv2d(6, 16, (3, 3), 1, 1, bias=False),
                                    #   BatchNorm2d(16, track_running_stats=False), 
                                      BatchNorm2d(16, track_running_stats=True), 
                                    # norm_layer(16),
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

        feat4 = F.interpolate(feat4,
                                                size=(64, 64),
                                                mode='bilinear')
        dfea1 = self.dconv_layer1(torch.cat((feat4, feat3), 1))
        dfea1 = F.interpolate(dfea1,
                                                size=(128, 128),
                                                mode='bilinear')
        dfea2 = self.dconv_layer2(torch.cat((dfea1, feat2), 1))
        dfea2 = F.interpolate(dfea2,
                                                size=(256, 256),
                                                mode='bilinear')
        dfea3 = self.dconv_layer3(torch.cat((dfea2, feat1), 1))

        res_aligned = dfea3

        return res_aligned

