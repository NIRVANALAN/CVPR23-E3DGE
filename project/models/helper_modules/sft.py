'''
VQGAN code, adapted from the original created by the Unleashing Transformers authors:
https://github.com/samb-t/unleashing-transformers/blob/master/models/vqgan.py
'''
import torch
import torch.nn as nn

from project.models.helper_modules.resnetfc import ResnetBlockFC


def normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32,
                              num_channels=in_channels,
                              eps=1e-6,
                              affine=True)


class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels=None):
        super(ResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.norm1 = normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels,
                               out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.norm2 = normalize(out_channels)
        self.conv2 = nn.Conv2d(out_channels,
                               out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        if self.in_channels != self.out_channels:
            self.conv_out = nn.Conv2d(in_channels,
                                      out_channels,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0)

    def forward(self, x_in):
        x = x_in
        x = self.norm1(x)
        x = swish(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = swish(x)
        x = self.conv2(x)
        if self.in_channels != self.out_channels:
            x_in = self.conv_out(x_in)

        return x + x_in


# https://github.com/sczhou/CodeFormer/blob/master/basicsr/archs/codeformer_arch.py
class Fuse_sft_block(nn.Module):

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.encode_enc = ResBlock(2 * in_ch, out_ch)

        self.scale = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1))

        self.shift = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1))

    def forward(self, enc_feat, dec_feat, w=1):
        enc_feat = self.encode_enc(torch.cat([enc_feat, dec_feat], dim=1))
        scale = self.scale(enc_feat)
        shift = self.shift(enc_feat)
        residual = w * (dec_feat * scale + shift)
        out = dec_feat + residual
        return out


# MLP based
class Fuse_sft_MLP(nn.Module):

    def __init__(self, in_ch=256 + 1, out_ch=256):
        super().__init__()
        # self.encode_enc = ResBlock(2*in_ch, out_ch)
        self.encode_enc = ResnetBlockFC(in_ch + out_ch, out_ch)

        self.scale = nn.Sequential(
            nn.Linear(out_ch, out_ch),
            nn.LeakyReLU(0.2, True),
            nn.Linear(out_ch, out_ch),
        )

        self.shift = nn.Sequential(
            nn.Linear(out_ch, out_ch),
            nn.LeakyReLU(0.2, True),
            nn.Linear(out_ch, out_ch),
        )

    def forward(self, enc_feat, dec_feat, w=1):
        enc_feat = self.encode_enc(torch.cat([enc_feat, dec_feat], dim=-1))
        scale = self.scale(enc_feat)
        shift = self.shift(enc_feat)
        residual = w * (dec_feat * scale + shift)
        out = dec_feat + residual
        return out
