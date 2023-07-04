import math
from enum import Enum

import numpy as np
import torch
from project.models.helper_modules.helpers import (_upsample_add, bottleneck_IR,
                                    bottleneck_IR_SE, get_blocks)
from project.models.helper_modules.helpers import GradualStyleBlock
from project.models.stylesdf_model import EqualLinear
from torch import nn
from torch.nn import BatchNorm2d, Conv2d, Module, PReLU, Sequential


class ProgressiveStage(Enum):
    WTraining = 0
    Delta1Training = 1
    Delta2Training = 2
    Delta3Training = 3
    Delta4Training = 4
    Delta5Training = 5
    Delta6Training = 6
    Delta7Training = 7
    Delta8Training = 8
    Delta9Training = 9
    Delta10Training = 10
    Delta11Training = 11
    Delta12Training = 12
    Delta13Training = 13
    Delta14Training = 14
    Delta15Training = 15
    Delta16Training = 16
    Delta17Training = 17
    Inference = 18


class Encoder4Editing(Module):

    def __init__(self, num_layers, mode='ir', opts=None):
        super(Encoder4Editing, self).__init__()
        assert num_layers in [50, 100,
                              152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64), PReLU(64))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(bottleneck.in_channel, bottleneck.depth,
                                bottleneck.stride))
        self.body = Sequential(*modules)

        self.styles = nn.ModuleList()
        log_size = int(math.log(opts.stylegan_size, 2))
        self.style_count = 2 * log_size - 2
        self.coarse_ind = 3
        self.middle_ind = 7

        for i in range(self.style_count):
            if i < self.coarse_ind:
                style = GradualStyleBlock(512, 512, 16)
            elif i < self.middle_ind:
                style = GradualStyleBlock(512, 512, 32)
            else:
                style = GradualStyleBlock(512, 512, 64)
            self.styles.append(style)

        self.latlayer1 = nn.Conv2d(256,
                                   512,
                                   kernel_size=1,
                                   stride=1,
                                   padding=0)
        self.latlayer2 = nn.Conv2d(128,
                                   512,
                                   kernel_size=1,
                                   stride=1,
                                   padding=0)

        self.progressive_stage = ProgressiveStage.Inference

    def get_deltas_starting_dimensions(self):
        ''' Get a list of the initial dimension of every delta from which it is applied '''
        return list(range(
            self.style_count))  # Each dimension has a delta applied to it

    def set_progressive_stage(self, new_stage: ProgressiveStage):
        self.progressive_stage = new_stage
        print('Changed progressive stage to: ', new_stage)

    def forward(self, x):
        x = self.input_layer(x)

        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 6:
                c1 = x
            elif i == 20:
                c2 = x
            elif i == 23:
                c3 = x

        # Infer main W and duplicate it
        w0 = self.styles[0](c3)
        w = w0.repeat(self.style_count, 1, 1).permute(1, 0, 2)
        stage = self.progressive_stage.value
        features = c3
        for i in range(1, min(stage + 1,
                              self.style_count)):  # Infer additional deltas
            if i == self.coarse_ind:
                p2 = _upsample_add(c3,
                                   self.latlayer1(c2))  # FPN's middle features
                features = p2
            elif i == self.middle_ind:
                p1 = _upsample_add(p2,
                                   self.latlayer2(c1))  # FPN's fine features
                features = p1
            delta_i = self.styles[i](features)
            w[:, i] += delta_i
        return w


class GradualStyleEncoder(Module):

    def __init__(self, num_layers, mode='ir', opts=None):
        super(GradualStyleEncoder, self).__init__()
        assert num_layers in [50, 100,
                              152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64), PReLU(64))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(bottleneck.in_channel, bottleneck.depth,
                                bottleneck.stride))
        self.body = Sequential(*modules)

        self.styles = nn.ModuleList()
        log_size = int(math.log(opts.stylegan_size, 2))
        self.style_count = 2 * log_size - 2
        self.coarse_ind = 3
        self.middle_ind = 7
        for i in range(self.style_count):
            if i < self.coarse_ind:
                style = GradualStyleBlock(512, 512, 16)
            elif i < self.middle_ind:
                style = GradualStyleBlock(512, 512, 32)
            else:
                style = GradualStyleBlock(512, 512, 64)
            self.styles.append(style)
        self.latlayer1 = nn.Conv2d(256,
                                   512,
                                   kernel_size=1,
                                   stride=1,
                                   padding=0)
        self.latlayer2 = nn.Conv2d(128,
                                   512,
                                   kernel_size=1,
                                   stride=1,
                                   padding=0)

    def forward(self, x):
        x = self.input_layer(x)

        latents = []
        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 6:
                c1 = x
            elif i == 20:
                c2 = x
            elif i == 23:
                c3 = x

        for j in range(self.coarse_ind):
            latents.append(self.styles[j](c3))

        p2 = _upsample_add(c3, self.latlayer1(c2))
        for j in range(self.coarse_ind, self.middle_ind):
            latents.append(self.styles[j](p2))

        p1 = _upsample_add(p2, self.latlayer2(c1))
        for j in range(self.middle_ind, self.style_count):
            latents.append(self.styles[j](p1))

        out = torch.stack(latents, dim=1)
        return out


class BackboneEncoderUsingLastLayerIntoW(Module):

    def __init__(self, num_layers, mode='ir', opts=None):
        super(BackboneEncoderUsingLastLayerIntoW, self).__init__()
        print('Using BackboneEncoderUsingLastLayerIntoW')
        assert num_layers in [50, 100,
                              152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64), PReLU(64))
        self.output_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.linear = EqualLinear(512, 512, lr_mul=1)
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(bottleneck.in_channel, bottleneck.depth,
                                bottleneck.stride))
        self.body = Sequential(*modules)
        log_size = int(math.log(opts.stylegan_size, 2))
        self.style_count = 2 * log_size - 2

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_pool(x)
        x = x.reshape(-1, 512)
        x = self.linear(x)
        return x.repeat(self.style_count, 1, 1).permute(1, 0, 2)


class Encoder4EditingHybrid(Module):

    def __init__(self, num_layers, mode='ir', opts=None):
        super().__init__()
        assert num_layers in [50, 100,
                              152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64), PReLU(64))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(bottleneck.in_channel, bottleneck.depth,
                                bottleneck.stride))
        self.body = Sequential(*modules)

        self.styles = nn.ModuleList()

        self.style_count = 10
        self.coarse_ind = 3
        self.middle_ind = 7

        # * 6 resolutions in total
        self.pigan_coarse_ind = 3  # 16
        self.pigan_middle_ind = 6  # 32
        self.pigan_style_count = 9  # 64

        for i in range(9):  # pigan
            if i < self.pigan_coarse_ind:
                style = GradualStyleBlock(512, 256, 16)  # todo
            elif i < self.pigan_middle_ind:
                style = GradualStyleBlock(512, 256, 32)
            else:
                style = GradualStyleBlock(512, 256, 64)
            self.styles_pigan.append(style)

        for i in range(self.style_count):
            if i < self.coarse_ind:
                style = GradualStyleBlock(512, 512, 16)
            elif i < self.middle_ind:
                style = GradualStyleBlock(512, 512, 32)
            else:
                style = GradualStyleBlock(512, 512, 64)
            self.styles.append(style)

        self.latlayer1 = nn.Conv2d(256,
                                   512,
                                   kernel_size=1,
                                   stride=1,
                                   padding=0)
        self.latlayer2 = nn.Conv2d(128,
                                   512,
                                   kernel_size=1,
                                   stride=1,
                                   padding=0)
        self.latlayer3 = nn.Conv2d(64, 512, kernel_size=1, stride=1, padding=0)

        self.progressive_stage = ProgressiveStage.Inference

    def get_deltas_starting_dimensions(self):
        ''' Get a list of the initial dimension of every delta from which it is applied '''
        return list(range(
            self.style_count))  # Each dimension has a delta applied to it

    def set_progressive_stage(self, new_stage: ProgressiveStage):
        self.progressive_stage = new_stage
        print('Changed progressive stage to: ', new_stage)

    def forward(self, x):
        x = self.input_layer(x)
        c256 = x  # 64; 256

        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 2:
                c0 = x  # c128
            if i == 6:
                c1 = x  # c64
            elif i == 20:
                c2 = x  # c32
            elif i == 23:
                c3 = x  # c16

        latents = []

        features = c3
        w_thumb = self.styles[0](c3)
        for i in range(1, min(stage + 1, self.pigan_style_count)):
            if i == self.coarse_ind:
                p2 = _upsample_add(c3,
                                   self.latlayer1(c2))  # FPN's middle features
                features = p2
            elif i == self.middle_ind:
                p1 = _upsample_add(p2,
                                   self.latlayer2(c1))  # FPN's fine features
                features = p1
            delta_i = self.styles[i](features)
            w_thumb[:, i] += delta_i

        for j in range(self.pigan_coarse_ind):
            latents.append(self.styles_pigan[j](c3))

        p32 = self._upsample_add(c3, self.latlayer256(c2))
        for j in range(self.pigan_coarse_ind, self.pigan_middle_ind):
            latents.append(self.styles_pigan[j](p32))

        p64 = self._upsample_add(p32, self.latlayer128(c1))
        for j in range(self.pigan_middle_ind, self.pigan_style_count):
            latents.append(self.styles_pigan[j](p64))
        w_thumb = torch.stack(latents, dim=1)

        # Infer main W and duplicate it
        w0 = self.styles[0](c3)
        w_stylegan = w0.repeat(self.style_count, 1, 1).permute(1, 0, 2)
        stage = self.progressive_stage.value
        features = c3
        for i in range(1, min(stage + 1,
                              self.style_count)):  # Infer additional deltas
            if i == self.coarse_ind:
                p2 = _upsample_add(c3,
                                   self.latlayer1(c2))  # FPN's middle features
                features = p2
            elif i == self.middle_ind:
                p1 = _upsample_add(p2,
                                   self.latlayer2(c1))  # FPN's fine features
                features = p1
            delta_i = self.styles[i](features)
            w_stylegan[:, i] += delta_i
        # return w
        return [w_thumb, w_stylegan]


class Encoder4EditingHybridBaseline(Module):

    def __init__(self, num_layers, mode='ir', opts=None):
        super().__init__()
        assert num_layers in [50, 100,
                              152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64), PReLU(64))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(bottleneck.in_channel, bottleneck.depth,
                                bottleneck.stride))
        self.body = Sequential(*modules)

        self.styles_pigan = nn.ModuleList()
        self.styles = nn.ModuleList()
        self.pigan_style_count = 9
        # self.style_count = 10
        self.coarse_ind = 3
        self.middle_ind = 7
        self.style_count = 10

        self.pigan_coarse_indx = 6

        # * pigan
        for i in range(self.pigan_style_count):  # * 6:3
            if i < self.pigan_coarse_indx:
                style = GradualStyleBlock(512, 256, 16)  # todo
            else:
                style = GradualStyleBlock(512, 256, 32)
            self.styles_pigan.append(style)

        # * styelgan
        for i in range(self.style_count):
            if i < self.coarse_ind:
                style = GradualStyleBlock(512, 512, 16)
            elif i < self.middle_ind:
                style = GradualStyleBlock(512, 512, 32)
            else:
                style = GradualStyleBlock(512, 512, 64)
            self.styles.append(style)

        self.latlayer1 = nn.Conv2d(256,
                                   512,
                                   kernel_size=1,
                                   stride=1,
                                   padding=0)
        self.latlayer2 = nn.Conv2d(128,
                                   512,
                                   kernel_size=1,
                                   stride=1,
                                   padding=0)
        self.latlayer3 = nn.Conv2d(64, 512, kernel_size=1, stride=1, padding=0)

        self.progressive_stage = ProgressiveStage.Inference

    def get_deltas_starting_dimensions(self):
        ''' Get a list of the initial dimension of every delta from which it is applied '''
        return list(range(
            self.style_count))  # Each dimension has a delta applied to it

    def set_progressive_stage(self, new_stage: ProgressiveStage):
        self.progressive_stage = new_stage
        print('Changed progressive stage to: ', new_stage)

    def forward(self, x):
        x = self.input_layer(x)
        # c256 = x  # 64; 256

        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            # if i == 2:
            #     c0 = x # c128
            if i == 6:
                c1 = x  # c64
            elif i == 20:
                c2 = x  # c32
            elif i == 23:
                c3 = x  # c16

        stage = self.progressive_stage.value

        features = c3
        w_thumb = self.styles_pigan[0](c3)  # 4 256
        w_thumb = w_thumb.repeat(self.pigan_style_count, 1, 1).permute(1, 0, 2)
        for i in range(1, min(stage + 1, self.pigan_style_count)):
            if i == self.pigan_coarse_indx:
                p2 = _upsample_add(c3,
                                   self.latlayer1(c2))  # FPN's middle features
                features = p2
            delta_i = self.styles_pigan[i](features)
            w_thumb[:, i] += delta_i

        w0 = self.styles[0](c3)
        w_stylegan = w0.repeat(self.style_count, 1, 1).permute(1, 0, 2)
        features = c3
        for i in range(1, min(stage + 1,
                              self.style_count)):  # Infer additional deltas
            if i == self.coarse_ind:
                p2 = _upsample_add(c3,
                                   self.latlayer1(c2))  # FPN's middle features
                features = p2
            elif i == self.middle_ind:
                p1 = _upsample_add(p2,
                                   self.latlayer2(c1))  # FPN's fine features
                features = p1
            delta_i = self.styles[i](features)
            w_stylegan[:, i] += delta_i
        # return w
        # ipdb.set_trace()
        return [w_thumb, w_stylegan]
