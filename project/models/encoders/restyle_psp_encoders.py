# from models.stylesdf_model import VolumeRenderDiscriminatorEncoder, VolumeRenderDiscriminator, Discriminator
# ipdb.set_trace()

import torch
from torch import nn
from torch.nn import Conv2d, BatchNorm2d, PReLU, Sequential, Module
from torchvision.models.resnet import resnet34

from ..helper_modules.helpers import get_blocks, bottleneck_IR, bottleneck_IR_SE, GradualStyleBlock
# from ....deprecated.map2style import GradualStyleBlock


class BackboneEncoder(Module):
    """
    The simpler backbone architecture used by ReStyle where all style vectors are extracted from the final 16x16 feature
    map of the encoder. This classes uses the simplified architecture applied over an ResNet IRSE-50 backbone.
    Note this class is designed to be used for the human facial domain.
    """

    def __init__(self, num_layers, mode='ir', n_styles=18, opts=None):
        super(BackboneEncoder, self).__init__()
        assert num_layers in [50, 100,
                              152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE

        self.input_layer = Sequential(
            Conv2d(opts.input_nc, 64, (3, 3), 1, 1, bias=False),
            BatchNorm2d(64), PReLU(64))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(bottleneck.in_channel, bottleneck.depth,
                                bottleneck.stride))
        self.body = Sequential(*modules)

        self.styles = nn.ModuleList()
        self.style_count = n_styles
        for i in range(self.style_count):
            style = GradualStyleBlock(512, 512, 16)
            self.styles.append(style)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        latents = []
        for j in range(self.style_count):
            latents.append(self.styles[j](x))
        out = torch.stack(latents, dim=1)
        return out


class HybridBackboneEncoder(BackboneEncoder):

    def __init__(self,
                 num_layers,
                 mode='ir',
                 n_styles=10,
                 n_thumb_styles=9,
                 opts=None):
        super().__init__(num_layers, mode='ir', n_styles=10, opts=opts)
        self.thumb_style_count = n_thumb_styles
        self.thumb_styles = nn.ModuleList()

        for i in range(self.thumb_style_count):
            thumb_style = GradualStyleBlock(512, 256, 16)
            self.thumb_styles.append(thumb_style)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)

        thumb_latents = []
        for j in range(self.thumb_style_count):
            thumb_latents.append(self.thumb_styles[j](x))
        out_thumb = torch.stack(thumb_latents, dim=1)

        latents = []
        for j in range(self.style_count):
            latents.append(self.styles[j](x))
        out = torch.stack(latents, dim=1)

        return [out_thumb, out]  # B,9,256 B,10,512


class ResNetBackboneEncoder(Module):
    """
    The simpler backbone architecture used by ReStyle where all style vectors are extracted from the final 16x16 feature
    map of the encoder. This classes uses the simplified architecture applied over an ResNet34 backbone.
    """

    def __init__(self, n_styles=18, opts=None):
        super(ResNetBackboneEncoder, self).__init__()

        self.conv1 = nn.Conv2d(opts.input_nc,
                               64,
                               kernel_size=7,
                               stride=2,
                               padding=3,
                               bias=False)
        self.bn1 = BatchNorm2d(64)
        self.relu = PReLU(64)

        resnet_basenet = resnet34(pretrained=True)
        blocks = [
            resnet_basenet.layer1, resnet_basenet.layer2,
            resnet_basenet.layer3, resnet_basenet.layer4
        ]
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(bottleneck)
        self.body = Sequential(*modules)

        self.styles = nn.ModuleList()
        self.style_count = n_styles
        for i in range(self.style_count):
            style = GradualStyleBlock(512, 512, 16)
            self.styles.append(style)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.body(x)
        latents = []
        for j in range(self.style_count):
            latents.append(self.styles[j](x))
        out = torch.stack(latents, dim=1)
        return out


class BackboneEncoderRenderer(Module):
    """
    The simpler backbone architecture used by ReStyle where all style vectors are extracted from the final 16x16 feature
    map of the encoder. This classes uses the simplified architecture applied over an ResNet IRSE-50 backbone.
    Note this class is designed to be used for the human facial domain.
    """

    def __init__(self, num_layers, mode='ir', n_styles=9, opts=None):
        super().__init__()
        assert num_layers in [50, 100,
                              152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE

        self.input_layer = Sequential(
            Conv2d(opts.input_nc, 64, (3, 3), 1, 1, bias=False),
            BatchNorm2d(64), PReLU(64))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(bottleneck.in_channel, bottleneck.depth,
                                bottleneck.stride))
        self.body = Sequential(*modules)

        self.styles = nn.ModuleList()
        self.style_count = 2
        assert n_styles >= 2

        self.styles.append(GradualStyleBlock(512, 256, 16))
        self.styles.append(GradualStyleBlock(512, 512, 16))

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        latents = []
        for j in range(self.style_count):
            latents.append(self.styles[j](x))

        # if len(latents) == 1:  # w
        # out = latents[0]

        return [
            latents[0].unsqueeze(1).repeat_interleave(9, 1),
            latents[1].unsqueeze(1).repeat_interleave(10, 1)
        ]

        # else:  # w+
        #     out = torch.stack(latents, dim=1)

        # return [latents[0].unsqueeze(1).repeat_interleave(9, 1),
        #     latents[1].unsqueeze(1).repeat_interleave(10, 1)]
