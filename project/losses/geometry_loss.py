import torch
# from chamferdist import ChamferDistance

import torch
import torch.nn as nn


class Loss:

    def __init__(self, keys):
        """
        keys are used in multi-gpu model, DummyLoss in train_tools.py
        :param keys: the output keys of the dict
        """
        self.keys = keys

    def __call__(self, data_pr, data_gt, step, **kwargs):
        pass


class ConsistencyLoss(Loss):
    default_cfg = {
        'use_ray_mask': False,
        'use_dr_loss': False,
        'use_dr_fine_loss': False,
        'use_nr_fine_loss': False,
    }

    def __init__(self, cfg):
        self.cfg = {**self.default_cfg, **cfg}
        super().__init__([f'loss_prob', 'loss_prob_fine'])

    def __call__(self, hit_prob_pred, hit_prob_gt, **kwargs):
        # hit_prob, gt and pred loss.
        hit_prob_gt.detach_()
        # ! detach here

        prob0 = hit_prob_gt  # B H W Steps 1
        prob1 = hit_prob_pred
        # if self.cfg['use_ray_mask']:
        #     ray_mask = data_pr['ray_mask'].float()  # 1,rn
        # else:
        #     ray_mask = 1

        # st()
        # * Cross entropy. BCE here.
        bce = -prob0 * torch.log(prob1 + 1e-5) - (1 - prob0) * torch.log(
            1 - prob1 + 1e-5)  # B H W S 1
        # outputs = {'loss_prob': torch.mean(torch.mean(ce, -1), 1)}
        loss_prob = torch.mean(torch.mean(
            bce, -2))  # follow neuRay implementation for now. BCE.

        return loss_prob


class DepthLoss(Loss):
    default_cfg = {
        'depth_correct_thresh': 0.02,
        'depth_loss_type': 'l2',
        'depth_loss_l1_beta': 0.05,
    }

    def __init__(self, cfg):
        super().__init__(['loss_depth'])
        self.cfg = {**self.default_cfg, **cfg}
        if self.cfg['depth_loss_type'] == 'smooth_l1':
            self.loss_op = nn.SmoothL1Loss(reduction='none',
                                           beta=self.cfg['depth_loss_l1_beta'])

    def __call__(self, data_pr, ref_imgs_info, **kwargs):

        depth_pr = data_pr['depth_mean']  # rfn,pn
        depth_gt = ref_imgs_info['depth']  # rfn,1,h,w
        # ! detach gt depth here
        # rfn, _, h, w = depth_maps.shape
        # depth_gt = interpolate_feats(
        #     depth_maps,coords,h,w,padding_mode='border',align_corners=True)[...,0]   # rfn,pn

        # transform to inverse depth coordinate
        # near, far = -1/depth_range[:,0:1], -1/depth_range[:,1:2] # rfn,1

        # todo, use disparity space.
        depth_range = ref_imgs_info['depth_range']  # rfn,2
        near, far = depth_range[..., 0:1], depth_range[..., 1:2]  # N H W 1 2

        def process(depth):
            depth = torch.clamp(depth, min=1e-5)
            # depth = -1 / depth
            depth = (depth - near) / (far - near)
            depth = torch.clamp(depth, min=0, max=1.0)
            return depth

        # depth_gt = process(depth_gt)

        # compute loss
        def compute_loss(depth_pr):

            if self.cfg['depth_loss_type'] == 'l2':
                loss = (depth_gt - depth_pr)**2
            elif self.cfg['depth_loss_type'] == 'smooth_l1':
                loss = self.loss_op(depth_gt, depth_pr)

            loss = loss.mean()
            return loss

        # outputs = {'loss_depth': compute_loss(depth_pr)}
        loss_depth = compute_loss(depth_pr)

        return loss_depth
