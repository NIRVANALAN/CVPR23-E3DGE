import torch
import torch.nn.functional as F
from project.models.helper_modules.helpers import (bottleneck_IR, bottleneck_IR_SE,
                                    get_blocks)
from project.models.helper_modules.helpers import GradualStyleBlock
from project.utils.transform import gt_pool
from torch import nn
from torch.nn import BatchNorm2d, Conv2d, Module, PReLU, Sequential
from torchvision.models.resnet import resnet34


class GradualStyleEncoder(Module):
    """
    Original encoder architecture from pixel2style2pixel. 
    This classes uses an FPN-based architecture applied over an ResNet IRSE-50 backbone.
    Note this class is designed to be used for the human facial domain.
    """

    def __init__(self, num_layers, mode='ir', n_styles=18, opts=None):
        super(GradualStyleEncoder, self).__init__()
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
        self.coarse_ind = 3  # 2^2 - 2^4
        self.middle_ind = 7  # 2^5 - 2 ^ 8 (256)
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

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(
            x, size=(H, W), mode='bilinear', align_corners=True) + y

    def forward(self, x):
        x = self.input_layer(x)

        latents = []
        modulelist = list(self.body._modules.values())
        for i, l in enumerate(
                modulelist):  # * extract feature-maps of different resolution
            x = l(x)
            if i == 6:
                c1 = x
            elif i == 20:
                c2 = x
            elif i == 23:
                c3 = x

        for j in range(self.coarse_ind):
            latents.append(self.styles[j](c3))

        p2 = self._upsample_add(c3, self.latlayer1(c2))
        for j in range(self.coarse_ind, self.middle_ind):
            latents.append(self.styles[j](p2))

        p1 = self._upsample_add(p2, self.latlayer2(c1))
        for j in range(self.middle_ind, self.style_count):
            latents.append(self.styles[j](p1))

        out = torch.stack(latents, dim=1)
        return out


class HybridGradualStyleEncoder(Module):
    # * originally designed for 256 resolution
    # * for vol_renderer use only; first test this over 64 resolution
    def __init__(self, num_layers, mode='ir', n_styles=9 + 6, opts=None):
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
        self.full_pipeline = opts.full_pipeline
        self.body = Sequential(*modules)
        self.styles_pigan = nn.ModuleList()

        # * 6 resolutions in total
        self.pigan_coarse_ind = 3  # 16
        self.pigan_middle_ind = 6  # 32
        self.pigan_style_count = 9  # 64
        self.stylegan_style_count = 10  # 256

        for i in range(9):  # pigan
            if i < self.pigan_coarse_ind:
                style = GradualStyleBlock(512, 256, 16)  # todo
            elif i < self.pigan_middle_ind:
                style = GradualStyleBlock(512, 256, 32)
            else:
                style = GradualStyleBlock(512, 256, 64)
            self.styles_pigan.append(style)

        self.opts = opts
        if self.full_pipeline:
            self.stylegan_style_count = 10  # 256
            self.styles_stylegan = nn.ModuleList()

            # self.stylegan_coarse_ind = 2  # 64
            # self.stylegan_middle_ind = 4  # 128
            # self.stylegan_style_count = 6  # 256
            # * -=4(truncate)
            # * follow PsP
            # self.stylegan_coarse_ind = -1  # 64
            if not opts.single_decoder_layer:
                self.stylegan_coarse_ind = 0  # 64
                self.stylegan_middle_ind = 3  # 128
                # self.coarse_ind = -1 # 2^2 - 2^4
                # self.middle_ind = 3 # 2^5 - 2 ^ 8 (256)
                # self.middle_ind = 7
                for i in range(self.stylegan_style_count):  # stylegan
                    if i < self.stylegan_coarse_ind:  #
                        style = GradualStyleBlock(512, 512, 64)  # todo
                    elif i < self.stylegan_middle_ind:
                        style = GradualStyleBlock(512, 512, 128)
                    else:
                        style = GradualStyleBlock(512, 512, 256)
                    self.styles_stylegan.append(style)
            else:
                style = GradualStyleBlock(512, 512, 128)
                self.styles_stylegan.append(style)

            self.latlayer64 = nn.Conv2d(
                64,  # spatial 256, 128
                512,
                kernel_size=1,
                stride=1,
                padding=0)

        # * upsample channel
        # self.latlayer64 = nn.Conv2d(
        #     64,  # spatial 256, 128
        #     512,
        #     kernel_size=1,
        #     stride=1,
        #     padding=0)
        self.latlayer128 = nn.Conv2d(
            128,  # spatial 64
            512,
            kernel_size=1,
            stride=1,
            padding=0)
        self.latlayer256 = nn.Conv2d(
            256,  # spatial 32
            512,
            kernel_size=1,
            stride=1,
            padding=0)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(
            x, size=(H, W), mode='bilinear', align_corners=True) + y

    def forward(self, x):

        x = self.input_layer(x)
        c256 = x  # 64; 256

        modulelist = list(self.body._modules.values())
        for i, l in enumerate(
                modulelist):  # * extract feature-maps of different resolution
            x = l(x)
            # print(i, x.shape)
            if i == 2:
                c128 = x
            elif i == 6:
                c64 = x
            elif i == 20:
                c32 = x
            elif i == 23:
                c16 = x
        # 256 -> 16, 4 times downsample?

        # calculate pigan latents
        latents = []
        for j in range(self.pigan_coarse_ind):
            latents.append(self.styles_pigan[j](c16))

        p32 = self._upsample_add(c16, self.latlayer256(c32))
        for j in range(self.pigan_coarse_ind, self.pigan_middle_ind):
            latents.append(self.styles_pigan[j](p32))

        p64 = self._upsample_add(p32, self.latlayer128(c64))
        # p128 = self._upsample_add(p64, self.latlayer64(c128))
        for j in range(self.pigan_middle_ind, self.pigan_style_count):
            latents.append(self.styles_pigan[j](p64))
        thumb_out = torch.stack(latents, dim=1)

        if self.full_pipeline:
            p128 = self._upsample_add(p64, self.latlayer64(c128))
            # * calculate stylegan latents
            stylegan_latents = []
            if not self.opts.single_decoder_layer:
                for j in range(self.stylegan_coarse_ind):
                    stylegan_latents.append(self.styles_stylegan[j](p64))

                for j in range(self.stylegan_coarse_ind,
                               self.stylegan_middle_ind):
                    stylegan_latents.append(self.styles_stylegan[j](p128))

                p256 = self._upsample_add(p128, self.latlayer64(c256))
                for j in range(self.stylegan_middle_ind,
                               self.stylegan_style_count):
                    stylegan_latents.append(self.styles_stylegan[j](p256))
                stylegan_out = torch.stack(stylegan_latents, dim=1)
            else:
                stylegan_latents.append(self.styles_stylegan[0](p128))
                stylegan_out = stylegan_latents[0].unsqueeze(1).repeat(
                    1, self.stylegan_style_count, 1)

        else:
            stylegan_out = None

        return [thumb_out, stylegan_out]


class HybridGradualStyleEncoder_V2(Module):
    # * originally designed for 256 resolution
    # * for vol_renderer use only; first test this over 64 resolution
    def __init__(self, num_layers, mode, n_styles, opts):
        super().__init__()

        assert num_layers in [50, 100,
                              152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE

        self.opts = opts
        self.input_layer = Sequential(
            Conv2d(opts.input_nc, 64, (3, 3), 1, 1, bias=False),
            BatchNorm2d(64), PReLU(64))
        modules = []

        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(bottleneck.in_channel, bottleneck.depth,
                                bottleneck.stride))
        self.full_pipeline = opts.full_pipeline
        self.body = Sequential(*modules)
        self.styles_pigan = nn.ModuleList()

        # * 6 resolutions in total
        # self.pigan_coarse_ind = 4  # 16
        # self.pigan_middle_ind = 7  # 32
        # self.pigan_style_count = 9  # 64

        self.pigan_geo_layer = 6  # 16
        self.pigan_tex_layer = 9  # 32
        # self.pigan_style_count = 9  # 64

        for i in range(
                9
        ):  # pigan, first k(6) layers to model geometry, last 3 alyers to model texture
            if i < self.pigan_geo_layer:
                style = GradualStyleBlock(512, 256,
                                          opts.fpn_pigan_geo_layer_dim)  # todo
            elif i < self.pigan_tex_layer:
                style = GradualStyleBlock(512, 256,
                                          opts.fpn_pigan_tex_layer_dim)
            self.styles_pigan.append(style)

        # st()
        if self.full_pipeline and not opts.disable_decoder_fpn:
            self.enable_decoder = True
        else:
            self.enable_decoder = False

        if self.enable_decoder:
            self.styles_stylegan = nn.ModuleList()
            self.stylegan_style_count = 10  # 256
            # style = GradualStyleBlock(512, 512, opts.fpn_pigan_stylegan_layer_dim)  # todo

            if not opts.single_decoder_layer:
                self.stylegan_coarse_ind = 0  # 64
                self.stylegan_middle_ind = 3  # 128
                # self.coarse_ind = -1 # 2^2 - 2^4
                # self.middle_ind = 3 # 2^5 - 2 ^ 8 (256)
                # self.middle_ind = 7
                for i in range(self.stylegan_style_count):  # stylegan
                    if i < self.stylegan_coarse_ind:  #
                        style = GradualStyleBlock(512, 512, 64)  # todo
                    elif i < self.stylegan_middle_ind:
                        style = GradualStyleBlock(512, 512, 128)
                    else:
                        style = GradualStyleBlock(512, 512, 256)
                    self.styles_stylegan.append(style)
            else:
                style = GradualStyleBlock(512, 512, 128)
                self.styles_stylegan.append(style)

        # * upsample channel
        self.latlayer64 = nn.Conv2d(
            64,  # spatial 256, 128
            512,
            kernel_size=1,
            stride=1,
            padding=0)
        self.latlayer128 = nn.Conv2d(
            128,  # spatial 64
            512,
            kernel_size=1,
            stride=1,
            padding=0)
        self.latlayer256 = nn.Conv2d(
            256,  # spatial 32
            512,
            kernel_size=1,
            stride=1,
            padding=0)
        # self.gt_pool = gt_pool

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(
            x, size=(H, W), mode='bilinear', align_corners=True) + y

    def forward(self, x, return_featmap=False):
        if x.shape[-1] != 256:
            x = gt_pool(x)

        x = self.input_layer(x)

        # c256 = x  # 64; 256

        modulelist = list(self.body._modules.values())
        for i, l in enumerate(
                modulelist):  # * extract feature-maps of different resolution
            x = l(x)
            # print(i, x.shape)
            if i == 2:
                c128 = x
            elif i == 6:
                c64 = x
            elif i == 20:
                c32 = x
            elif i == 23:
                c16 = x
        # 256 -> 16, 4 times downsample?

        # calculate pigan latents
        latents = []
        # start pyramid from 32*32 feature maps

        p32 = self._upsample_add(c16, self.latlayer256(c32))  # * B*512*32*32
        p64 = self._upsample_add(p32, self.latlayer128(c64))
        # p256 = self._upsample_add(p128, self.latlayer64(c256))

        # ipdb.set_trace()
        for j in range(self.pigan_geo_layer):
            latents.append(self.styles_pigan[j](p32))

        for j in range(self.pigan_geo_layer, self.pigan_tex_layer):
            if self.opts.fpn_pigan_tex_layer_dim == 64:
                latents.append(self.styles_pigan[j](p64))
            else:
                latents.append(self.styles_pigan[j](p32))

        thumb_out = torch.stack(latents, dim=1)
        if self.enable_decoder:
            p128 = self._upsample_add(p64, self.latlayer64(c128))
            stylegan_latents = []

            stylegan_latents.append(self.styles_stylegan[0](p128))
            stylegan_out = stylegan_latents[0].unsqueeze(1).repeat(
                1, self.stylegan_style_count, 1)
            # todo
            if return_featmap:
                return {
                    'pred_latents': [thumb_out, stylegan_out],
                    'feat_maps': p64,
                    'p32': p32,
                    # 'p64': p64
                }
        else:
            stylegan_out = None

        return [thumb_out, stylegan_out]
        # return {'pred_latents':[thumb_out, stylegan_out] , 'feat_maps': p128}


class FPNwithLocalEncoder(HybridGradualStyleEncoder_V2):

    def __init__(self, num_layers, mode, opts, opt_local):
        super().__init__(num_layers, mode, opts)
        self.local_net = HGPIFuNetGAN(opt_local, 'projection')

    def forward(self, x):
        global_out = super().forward(x)
        local_out = self.local_net(x)
        # fusion


class ResNetGradualStyleEncoder(Module):
    """
    Original encoder architecture from pixel2style2pixel. This classes uses an FPN-based architecture applied over
    an ResNet34 backbone.
    """

    def __init__(self, n_styles=18, opts=None):
        super(ResNetGradualStyleEncoder, self).__init__()

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

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(
            x, size=(H, W), mode='bilinear', align_corners=True) + y

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        latents = []
        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 6:
                c1 = x  # * 128
            elif i == 12:
                c2 = x  # * 256
            elif i == 15:  # * 512;
                c3 = x

        for j in range(self.coarse_ind):
            latents.append(self.styles[j](c3))

        p2 = self._upsample_add(c3, self.latlayer1(c2))
        for j in range(self.coarse_ind, self.middle_ind):
            latents.append(self.styles[j](p2))

        p1 = self._upsample_add(p2, self.latlayer2(c1))
        for j in range(self.middle_ind, self.style_count):
            latents.append(self.styles[j](p1))

        out = torch.stack(latents, dim=1)
        return out
