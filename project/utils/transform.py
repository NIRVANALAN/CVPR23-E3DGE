import torch

thumb_pool = torch.nn.AdaptiveAvgPool2d((64, 64))
gt_pool = torch.nn.AdaptiveAvgPool2d((256, 256))