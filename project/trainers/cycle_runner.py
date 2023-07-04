import os
from copy import deepcopy

import torch
import torch.nn.functional as F
from project.utils import (
    requires_grad, )

from project.trainers.trainer import AERunner, RUNNER


@RUNNER.register_module(force=True)
class CycleRunner(AERunner):

    def __init__(
            self,
            encoder: torch.nn.Module,
            volume_discriminator: torch.nn.Module,
            generator: torch.nn.Module,
            mean_latent,
            opt: dict,
            device: torch.device,
            loaders: list,
            loss_class: torch.nn.Module = None,
            surface_g_ema: torch.nn.Module = None,
            ckpt=None,  # todo, to lint all params
            mode='train',
            work_dir=None,
            max_iters=None,
            max_epochs=None,
            discriminator=None):
        super().__init__(encoder, volume_discriminator, generator, mean_latent,
                         opt, device, loaders, loss_class, surface_g_ema, ckpt,
                         mode, work_dir, max_iters, max_epochs, discriminator)
        """Calculate the occlusion vis 3D projection
            1. get img feats 
            2. get depth
            3. encoder & decoder by MLP 
            4. weighted probability * residual as the target view map
        """

    def encode_ref_images(self,
                          images,
                          cam_settings=None,
                          no_editing=False,
                          editing_boundary_scale_list=None,
                          encoder_out=None):
        """encoder the images for all relevant infomration needed for query views

        Args:
            images (Tensor): reference view images
        """

        ref_imgs_info = dict(
            **self.encoder_ref_images_global(images, cam_settings, no_editing,
                                             editing_boundary_scale_list,
                                             encoder_out), )
        res_gt = ref_imgs_info['res_gt']

        if 'depth' in self.opt.rendering.residual_context_feats:
            depth = ref_imgs_info['depth']
            depth_as_feat = depth.permute(0, 3, 4, 1, 2).squeeze(1)  # B 1 H W

            if res_gt.shape != depth_as_feat.shape:
                depth_as_feat = F.interpolate(depth_as_feat,
                                              size=res_gt.shape[-2:])
        else:
            depth_as_feat = None

        # 2.2 get the E feature map
        if 'feat_maps' in self.opt.rendering.residual_context_feats:
            feat_maps = F.interpolate(feat_maps, size=res_gt.shape[-2:])
        else:
            feat_maps = None

        # ! 3. calculate ref view residual feature map
        self.g_module.renderer.network.netLocal.filter(
            res_gt,
            depth_as_feat,
            feat_maps,
            feat_key='ref_view',
        )

        # !
        ref_imgs_info.update(
            dict(
                img_feats=feat_maps,
                depth_feat=depth_as_feat,
            ))

        return ref_imgs_info

    def que_render_given_ref(self,
                             ref_imgs_info: dict,
                             que_info: dict,
                             self_rec=False,
                             im_feat_for_query=None,
                             **kwargs):
        """image based rendering of query viewpoint considering ref viewpoint
        3D version

        Args:
            ref_imgs_info (dict): from encoder_ref_images
            que_info (dict): information needed of a query viewpoint. from latent2image() or cycle training swapped batch

        Returns:
            dict: rendered information
        """
        # 1. query info needed
        que_cam_settings = que_info['cam_settings']
        ref_render_out = ref_imgs_info['global_render_out']

        # 2. get projected feats
        que_wd_pts = que_info['points']
        calibs = ref_imgs_info['cam_settings']['calibs']

        assert que_wd_pts.ndim == 5  # B H W S 3
        B, H, W, S, _ = que_wd_pts.shape

        # reshape for query feat
        que_wd_pts_B3N = que_wd_pts.clone().reshape(B, -1, 3).permute(
            0, 2, 1)  # B 3 N, for feat query

        # Get residual feats
        que_proj_ref_feats_BNC = self.g_module.renderer.network.netLocal.query(
            points=que_wd_pts_B3N,
            calibs=calibs,
            return_eikonal=False,
            return_feat_only=True,
            im_feat=im_feat_for_query)  # B, 256, N
        in_img_mask = que_proj_ref_feats_BNC['in_img'].reshape(B, H, W, S, 1)

        if self_rec:
            hit_prob = ref_render_out['hit_prob']
        # v0: close the weighting
        elif self.opt.rendering.disable_ref_view_weight:
            hit_prob = torch.ones_like(hit_prob)
        else:  # enable ref view weighting
            if self.g_module.renderer.force_background:

                none_stop_points = que_wd_pts[
                    ..., :-1, :]  # ignore calculating the last point hit-porb

                none_stop_points_hit_prob = self.g_module.renderer.query_hitting_probability_fixed_interval(
                    none_stop_points, ref_imgs_info)  # B H W S 1

                stop_points_hit_prob = torch.ones(
                    B, H, W, 1, 1,
                    device=none_stop_points_hit_prob.device) - torch.sum(
                        none_stop_points_hit_prob,
                        self.g_module.renderer.samples_dim,
                        keepdim=True)
                hit_prob = torch.cat(
                    (none_stop_points_hit_prob, stop_points_hit_prob),
                    self.g_module.renderer.samples_dim)
            else:
                hit_prob = self.g_module.renderer.query_hitting_probability_fixed_interval(
                    que_wd_pts, ref_imgs_info)  # B H W S 1

        if not self.opt.rendering.disable_ref_view_mask:
            hit_prob *= in_img_mask  # B H W S C

        pix_aligned_residual_feats = que_proj_ref_feats_BNC['feats'].permute(
            0, 2, 1).reshape(B, H, W, S, -1)  # reshape from BCN

        pix_aligned_residual_feats = self._retrieve_pix_aligned_residual_feats(
            pix_aligned_residual_feats)

        # 5. high res reconstruct
        visible_local_condition = {
            'feats': pix_aligned_residual_feats,
        }  # image as input
        res_render_out = self.latent2image(
            ref_imgs_info['pred_latents'],
            que_cam_settings,
            local_condition=visible_local_condition)  # add later

        return dict(res_render_out=res_render_out,
                    hit_prob=hit_prob,
                    in_img_mask=in_img_mask)
