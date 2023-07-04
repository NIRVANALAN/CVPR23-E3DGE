from collections import namedtuple
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.nn import Conv2d, BatchNorm2d, PReLU, ReLU, Sigmoid, MaxPool2d, AdaptiveAvgPool2d, Sequential, Module
# from project.models.stylesdf_model import EqualLinear
"""
ArcFace implementation from [TreB1eN](https://github.com/TreB1eN/InsightFace_Pytorch)
"""


# from project.models.stylesdf_model import ModulatedConv2d
class DemodulatedConv2d(nn.Module):

    def __init__(self,
                 in_channel,
                 out_channel,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 bias=False,
                 dilation=1):
        super().__init__()
        # https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/issues/411. fix droplet issue

        self.eps = 1e-8

        if not isinstance(kernel_size, tuple):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        self.in_channel = in_channel
        self.out_channel = out_channel

        self.weight = nn.Parameter(
            # torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
            torch.randn(1, out_channel, in_channel, *kernel_size))
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channel))

        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, input):
        batch, in_channel, height, width = input.shape

        demod = torch.rsqrt(self.weight.pow(2).sum([2, 3, 4]) + 1e-8)
        demod = demod.repeat_interleave(batch, 0)
        weight = self.weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(
            # batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
            batch * self.out_channel,
            in_channel,
            *self.kernel_size)

        input = input.view(1, batch * in_channel, height, width)
        if self.bias is None:
            out = F.conv2d(input,
                           weight,
                           padding=self.padding,
                           groups=batch,
                           dilation=self.dilation,
                           stride=self.stride)
        else:
            out = F.conv2d(input,
                           weight,
                           bias=self.bias,
                           padding=self.padding,
                           groups=batch,
                           dilation=self.dilation,
                           stride=self.stride)
        _, _, height, width = out.shape
        out = out.view(batch, self.out_channel, height, width)

        return out


class Flatten(Module):

    def forward(self, input):
        return input.reshape(input.size(0), -1)


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    """ A named tuple describing a ResNet block. """


def get_block(in_channel, depth, num_units, stride=2):
    return [Bottleneck(in_channel, depth, stride)
            ] + [Bottleneck(depth, depth, 1) for i in range(num_units - 1)]


def get_blocks(num_layers):
    if num_layers == 50:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=14),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13),
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 152:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=8),
            get_block(in_channel=128, depth=256, num_units=36),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    else:
        raise ValueError(
            "Invalid number of layers: {}. Must be one of [50, 100, 152]".
            format(num_layers))
    return blocks


class SEModule(Module):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(channels,
                          channels // reduction,
                          kernel_size=1,
                          padding=0,
                          bias=False)
        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(channels // reduction,
                          channels,
                          kernel_size=1,
                          padding=0,
                          bias=False)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class bottleneck_IR(Module):

    def __init__(self,
                 in_channel,
                 depth,
                 stride,
                 norm_layer=None,
                 demodulate=False):
        super(bottleneck_IR, self).__init__()
        if norm_layer is None:
            norm_layer = BatchNorm2d
        if demodulate:
            conv2d = DemodulatedConv2d
        else:
            conv2d = Conv2d

        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                # Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                conv2d(in_channel, depth, (1, 1), stride, bias=False),
                norm_layer(depth))


# BatchNorm2d(depth)
        self.res_layer = Sequential(
            # BatchNorm2d(in_channel),
            norm_layer(in_channel),
            # Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            PReLU(depth),
            # Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            norm_layer(depth))
        # BatchNorm2d(depth))

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


class bottleneck_IR_SE(Module):

    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR_SE, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            PReLU(depth), Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            BatchNorm2d(depth), SEModule(depth, 16))

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


def _upsample_add(x, y):
    """Upsample and add two feature maps.
    Args:
      x: (Variable) top feature map to be upsampled.
      y: (Variable) lateral feature map.
    Returns:
      (Variable) added feature map.
    Note in PyTorch, when input size is odd, the upsampled feature map
    with `F.upsample(..., scale_factor=2, mode='nearest')`
    maybe not equal to the lateral feature map size.
    e.g.
    original input size: [N,_,15,15] ->
    conv2d feature map size: [N,_,8,8] ->
    upsampled feature map size: [N,_,16,16]
    So we choose bilinear upsample which supports arbitrary output sizes.
    """
    _, _, H, W = y.size()
    return F.interpolate(x, size=(H, W), mode='bilinear',
                         align_corners=True) + y


# from # copied from https://github.dev/liuyuan-pal/NeuRay


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=dilation,
                     groups=groups,
                     bias=False,
                     dilation=dilation,
                     padding_mode='reflect')


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False,
                     padding_mode='reflect')


def interpolate_feats(feats,
                      points,
                      h=None,
                      w=None,
                      padding_mode='zeros',
                      align_corners=False,
                      inter_mode='bilinear'):
    """

    :param feats:   b,f,h,w
    :param points:  b,n,2
    :param h:       float
    :param w:       float
    :param padding_mode:
    :param align_corners:
    :param inter_mode:
    :return:
    """
    b, _, ch, cw = feats.shape
    if h is None and w is None:
        h, w = ch, cw
    x_norm = points[:, :, 0] / (w - 1) * 2 - 1
    y_norm = points[:, :, 1] / (h - 1) * 2 - 1
    points_norm = torch.stack([x_norm, y_norm], -1).unsqueeze(1)  # [srn,1,n,2]
    feats_inter = F.grid_sample(feats,
                                points_norm,
                                mode=inter_mode,
                                padding_mode=padding_mode,
                                align_corners=align_corners).squeeze(
                                    2)  # srn,f,n
    feats_inter = feats_inter.permute(0, 2, 1)
    return feats_inter


def masked_mean_var(feats, mask, dim=2):
    mask = mask.float()  # b,1,n,1
    mask_sum = torch.clamp_min(torch.sum(mask, dim, keepdim=True),
                               min=1e-4)  # b,1,1,1
    feats_mean = torch.sum(feats * mask, dim,
                           keepdim=True) / mask_sum  # b,f,1,1
    feats_var = torch.sum((feats - feats_mean)**2 * mask, dim,
                          keepdim=True) / mask_sum  # b,f,1,1
    return feats_mean, feats_var


class ResidualBlock(nn.Module):

    def __init__(self,
                 dim_in,
                 dim_out,
                 dim_inter=None,
                 use_norm=True,
                 norm_layer=nn.BatchNorm2d,
                 bias=False):
        super().__init__()
        if dim_inter is None:
            dim_inter = dim_out

        if use_norm:
            self.conv = nn.Sequential(
                norm_layer(dim_in),
                nn.ReLU(True),
                nn.Conv2d(dim_in,
                          dim_inter,
                          3,
                          1,
                          1,
                          bias=bias,
                          padding_mode='reflect'),
                norm_layer(dim_inter),
                nn.ReLU(True),
                nn.Conv2d(dim_inter,
                          dim_out,
                          3,
                          1,
                          1,
                          bias=bias,
                          padding_mode='reflect'),
            )
        else:
            self.conv = nn.Sequential(
                nn.ReLU(True),
                nn.Conv2d(dim_in, dim_inter, 3, 1, 1),
                nn.ReLU(True),
                nn.Conv2d(dim_inter, dim_out, 3, 1, 1),
            )

        self.short_cut = None
        if dim_in != dim_out:
            self.short_cut = nn.Conv2d(dim_in, dim_out, 1, 1)

    def forward(self, feats):
        feats_out = self.conv(feats)
        if self.short_cut is not None:
            feats_out = self.short_cut(feats) + feats_out
        else:
            feats_out = feats_out + feats
        return feats_out


class AddBias(nn.Module):

    def __init__(self, val):
        super().__init__()
        self.val = val

    def forward(self, x):
        return x + self.val


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 groups=1,
                 base_width=64,
                 dilation=1,
                 norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes, track_running_stats=False, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes, track_running_stats=False, affine=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class conv(nn.Module):

    def __init__(self, num_in_layers, num_out_layers, kernel_size, stride, norm='in'):
        super(conv, self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(num_in_layers,
                              num_out_layers,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=(self.kernel_size - 1) // 2,
                              padding_mode='reflect')
        if norm == 'in': # default behaviour
            self.bn = nn.InstanceNorm2d(num_out_layers,
                                        track_running_stats=False,
                                        affine=True)
        else:
            self.bn = nn.BatchNorm2d(
                num_out_layers,
                affine=True,
                track_running_stats=True
            )

    def forward(self, x):
        return F.elu(self.bn(self.conv(x)), inplace=True)


class upconv(nn.Module):

    def __init__(self, num_in_layers, num_out_layers, kernel_size, scale):
        super(upconv, self).__init__()
        self.scale = scale
        self.conv = conv(num_in_layers, num_out_layers, kernel_size, 1)

    def forward(self, x):
        x = nn.functional.interpolate(x,
                                      scale_factor=self.scale,
                                      align_corners=True,
                                      mode='bilinear')
        return self.conv(x)

class GradualStyleBlock(Module):

    def __init__(self, in_c, out_c, spatial):
        from project.models.stylesdf_model import EqualLinear
        super(GradualStyleBlock, self).__init__()
        self.out_c = out_c
        self.spatial = spatial
        num_pools = int(np.log2(spatial))
        modules = []
        modules += [
            Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU()
        ]
        for i in range(num_pools - 1):
            modules += [
                Conv2d(out_c, out_c, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU()
            ]
        self.convs = nn.Sequential(*modules)
        self.linear = EqualLinear(out_c, out_c, lr_mul=1)

    def forward(self, x):
        x = self.convs(x)
        x = x.reshape(-1, self.out_c)
        x = self.linear(x)
        return x
