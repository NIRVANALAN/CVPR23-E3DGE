import torch
from torch.nn import Conv2d, BatchNorm2d, PReLU, Sequential, Module

from project.models.helper_modules.helpers import bottleneck_IR


# ADA
class ResidualAligner(Module):

    def __init__(self, opts=None):
        super(ResidualAligner, self).__init__()
        self.conv_layer1 = Sequential(Conv2d(6, 16, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(16), PReLU(16))

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
