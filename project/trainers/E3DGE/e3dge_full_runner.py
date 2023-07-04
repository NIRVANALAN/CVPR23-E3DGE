import os
from pathlib import Path
from pdb import set_trace as st

import mmcv
import torch
import torch.nn.functional as F
from project.models.helper_modules.alignment_old import FeatureAlignerBig
from project.utils import requires_grad
from project.utils.misc_utils import PosEncoding
from project.utils.training_utils import (_swap_odd_even_index_view,
                                          make_pair_same_noise)
from torchvision import utils

from project.trainers.base_runner import RUNNER

from .e3dge_2dalignonly_runner import E3DGE_2DAlignOnly_Runner
from project.models.helper_modules.sft import Fuse_sft_MLP


@RUNNER.register_module(force=True)
class E3DGE_Full_Runner(E3DGE_2DAlignOnly_Runner):
    """
    add 2D + 3D aligned implementation runner
    """

    def __init__(self,
                 encoder: torch.nn.Module,
                 volume_discriminator: torch.nn.Module,
                 generator: torch.nn.Module,
                 mean_latent,
                 opt: dict,
                 device: torch.device,
                 loaders: list,
                 loss_class: torch.nn.Module = None,
                 surface_g_ema: torch.nn.Module = None,
                 ckpt=None,
                 mode='train',
                 work_dir=None,
                 max_iters=None,
                 max_epochs=None,
                 discriminator=None):
        super().__init__(encoder,
                         volume_discriminator,
                         generator,
                         mean_latent,
                         opt,
                         device,
                         loaders,
                         loss_class,
                         surface_g_ema,
                         ckpt,
                         mode,
                         work_dir,
                         max_iters,
                         max_epochs,
                         discriminator=discriminator)
        """Calculate the occlusion vis 3D projection
            1. get img feats 
            2. get depth
            3. encoder & decoder by MLP 
            4. weighted probability * residual as the target view map
        """

        assert self.opt.training.enable_local_model
        assert self.opt.training.enable_G1
        if self.opt.rendering.residual_PE_type == 'coordinate':
            self.PE = PosEncoding(3, N_freqs=7)
        elif self.opt.rendering.residual_PE_type == 'depth':
            self.PE = PosEncoding(1, N_freqs=7)
        elif self.opt.rendering.residual_PE_type == 'None':
            self.PE = None
        else:
            self.PE = PosEncoding(3, N_freqs=1)
        self.add_vis_mask = True

    def encode_ref_images(self,
                          images,
                          cam_settings=None,
                          no_editing=False,
                          editing_boundary_scale_list=None,
                          encoder_out=None):
        """add 2D grid_align model editing version.
        encoder the images for all relevant infomration needed for query views

        Args:
            images (Tensor): reference view images
        """
        input_thumb_imgs = self.pool_64(images)
        input_imgs = self.pool_256(images)

        if encoder_out is None:  # could be reused when rendering video
            encoder_out = super().image2latents(input_imgs,
                                                return_featmap=True)
        if isinstance(encoder_out, dict):
            pred_latents = encoder_out['pred_latents']
        else:
            pred_latents = encoder_out

        # 2. get depth map
        if cam_settings is None:
            pred_cam_settings = self.image2camsettings(input_thumb_imgs)
        else:
            pred_cam_settings = cam_settings

        render_out = super().latent2image(
            pred_latents,
            pred_cam_settings,
            geometry_sample=None,  # just study pixel sup
            sample_with_renderer=True)  # add later

        # 1. residual
        render_out['gen_thumb_imgs_64'] = render_out['gen_thumb_imgs']
        render_out['gen_thumb_imgs'] = torch.nn.functional.interpolate(
            render_out['gen_thumb_imgs'], (256, 256))
        res_gt = (input_imgs - render_out['gen_thumb_imgs']).detach()

        orig_res_gt = res_gt.clone()

        # * process editing
        if self.opt.inference.editing_inference and not no_editing:
            if editing_boundary_scale_list is None:
                editing_boundary_scale_list = self.opt.inference.editing_boundary_scale
            edit_code_ret = self.edit_code(pred_latents,
                                           editing_boundary_scale_list)
            que_latents = edit_code_ret['edited_pred_latents']

            edit_render_out = self.latent2image(
                que_latents,
                pred_cam_settings,
                geometry_sample=None,  # just study pixel sup
                local_condition=None,
                sample_with_renderer=True,
                input_is_latent=True)  # add later

            edit_upsampled_thumb_imgs = torch.nn.functional.interpolate(
                edit_render_out['gen_thumb_imgs'], (256, 256))
            res_gt = self.grid_align(
                torch.cat((res_gt, edit_upsampled_thumb_imgs),
                          dim=1))  # use the high-res
            depth = edit_render_out['depth']  # B H W 1 1

        else:
            edit_render_out = None
            depth = render_out['depth']  # B H W 1 1
            que_latents = pred_latents

        if 'depth' in self.opt.rendering.residual_context_feats:
            depth_as_feat = depth.permute(0, 3, 4, 1, 2).squeeze(1)  # B 1 H W

            if res_gt.shape != depth_as_feat.shape:
                depth_as_feat = F.interpolate(depth_as_feat,
                                              size=res_gt.shape[-2:])
        else:
            depth_as_feat = None

        # 2.2 get the E feature map
        # ! check if OK when editing
        if 'feat_maps' in self.opt.rendering.residual_context_feats:
            feat_maps = F.interpolate(feat_maps, size=res_gt.shape[-2:])
            # E_feat_context = torch.cat((E_feat_context, feat_maps), 1)
        else:
            feat_maps = None

        # ! 3. calculate ref view residual feature map
        ref_view_aligned_feat = self.g_module.renderer.network.netLocal.filter(
            residual_images=res_gt,
            depth_feat=depth_as_feat,
            ref_feats=feat_maps,
            feat_key='ref_view',
            return_feat=True)[-1]

        ref_imgs_info = dict(ref_view_aligned_feat=ref_view_aligned_feat,
                             imgs=input_imgs,
                             cam_settings=pred_cam_settings,
                             orig_res_gt=orig_res_gt,
                             global_render_out=render_out,
                             res_gt=res_gt,
                             encoder_out=encoder_out,
                             edit_render_out=edit_render_out,
                             pred_latents=que_latents)

        return ref_imgs_info

    def que_render_given_ref(self,
                             ref_imgs_info: dict,
                             que_info: dict,
                             ref_view_aligned_feat=None,
                             **kwargs):
        """2D + 3D inpainter version.
        2D: align_inpaint_net(\pi_{Residual}(x), que_thumb_img) -> inpainted feats
        3D: f(PE(x), \pi_{Residual}(x)) -> 2*256 modulations.
        basically add PE.

        Args:
            ref_imgs_info (dict): from encoder_ref_images
            que_info (dict): information needed of a query viewpoint. from latent2image() or cycle training swapped batch

        Returns:
            dict: rendered information
        """
        # 1. query info needed
        res_gt = ref_imgs_info['orig_res_gt']
        que_cam_settings = que_info['cam_settings']

        # 2. get projected feats
        que_wd_pts = que_info['points']
        ref_calibs = ref_imgs_info['cam_settings']['calibs']

        que_cam_settings = que_info['cam_settings']
        que_calibs = que_info['cam_settings']['calibs']

        assert que_wd_pts.ndim == 5  # B H W S 3
        B, H, W, S, _ = que_wd_pts.shape
        que_wd_pts_B3N = que_wd_pts.clone().reshape(B, -1, 3).permute(
            0, 2, 1)  # B 3 N, for feat query

        # * 1. query 3D feature
        que_proj_ref_feats_BNC = self.g_module.renderer.network.netLocal.query(
            points=que_wd_pts_B3N,
            calibs=ref_calibs,  # ! reference calibs
            feat_key='ref_view',
            return_eikonal=False,
            return_feat_only=True,
            im_feat=ref_view_aligned_feat)  # B, 256, N
        in_img_mask = que_proj_ref_feats_BNC['in_img'].reshape(
            B, H, W, S, 1)  # ! to append
        feature_3dprojection = que_proj_ref_feats_BNC['feats'].permute(
            0, 2, 1).reshape(B, H, W, S, -1)  # reshape from BCN

        # * 2. get 2D features
        # * align with query view img + depth
        depth_as_feat = que_info['depth'].permute(0, 3, 4, 1,
                                                  2).squeeze(1)  # B 1 H W

        if res_gt.shape != depth_as_feat.shape:
            depth_as_feat = F.interpolate(depth_as_feat,
                                          size=res_gt.shape[-2:])

        # * 3 calculate visibilityy mask
        que_depth_wd_pts = que_info['xyz']  # B 3 H W
        que_depth_B3N = que_depth_wd_pts.reshape(B, 3, -1)

        # * proj ref view surface to the que view, check visibility & dc gt
        que_depth_to_ref_proj = self.g_module.renderer.network.netLocal.query(
            points=que_depth_B3N,
            calibs=ref_calibs,
            feat_key='none',
            return_eikonal=False,
            return_projection_only=True)  # B, 256, N
        que_in_ref_in_img_mask = que_depth_to_ref_proj['in_img'].reshape(
            B, H, W, 1, 1).repeat_interleave(S, -2)  # B H W S 1

        # 2.2 get the E feature map
        que_upsampled_thumb_imgs = torch.nn.functional.interpolate(
            que_info['gen_thumb_imgs'], (256, 256))
        aligned_res = self.grid_align(
            torch.cat((res_gt, que_upsampled_thumb_imgs),
                      dim=1))  # use the high-res
        residual_ctx_pe = self.PE(que_wd_pts)
        residual_ctx_pe = residual_ctx_pe.reshape(B, H, W, S, -1)

        # * filter aligned residual&depth
        que_view_2Daligned_E1feat = self.g_module.renderer.network.netLocal.filter(
            residual_images=aligned_res,
            depth_feat=depth_as_feat,
            ref_feats=None,
            feat_key='que_view',
            return_feat=True)[-1]

        que_2dproj_ref_feats_BNC = self.g_module.renderer.network.netLocal.query(
            points=que_wd_pts_B3N,
            calibs=que_calibs,  # !
            feat_key='que_view',
            return_eikonal=False,
            return_feat_only=True,
            im_feat=que_view_2Daligned_E1feat)  # * fix bug...

        feature_2dAlign = que_2dproj_ref_feats_BNC['feats'].permute(
            0, 2, 1).reshape(B, H, W, S, -1)  # reshape from BCN

        # ============ add PE of que points ============== #

        # ================ do 2D feature inpaint + 2D-3D hybrid feature alignment + reconstruction ==========#
        if self.add_vis_mask:  # * concat visibility mask here.
            feature_2dAlign = torch.cat(
                [feature_2dAlign, que_in_ref_in_img_mask], -1)

        # ================ do 2D feature inpaint + 2D-3D hybrid feature alignment ==========#
        pix_aligned_residual_feats = self.fuse_sft_block(
            feature_2dAlign, feature_3dprojection)

        # * add PE
        pix_aligned_residual_feats = torch.cat(
            (pix_aligned_residual_feats, residual_ctx_pe), -1)  # 256+45

        # 5. high res reconstruct
        visible_local_condition = {
            'feats': pix_aligned_residual_feats,
        }  # image as input

        # * ====== remove redundant computation
        res_render_out = self.latent2image(
            ref_imgs_info['pred_latents'],
            que_cam_settings,
            local_condition=visible_local_condition)

        que_img_for_vis = torch.cat((
            self.pool_256(res_gt),
            self.pool_256(aligned_res),
            self.pool_256(res_render_out['gen_imgs']),
        ), -1)

        return dict(res_render_out=res_render_out,
                    que_img_for_vis=que_img_for_vis,
                    aligned_res=aligned_res,
                    in_img_mask=in_img_mask)

    def _build_model(self):
        super()._build_model()
        self.fuse_sft_block = Fuse_sft_MLP(256 + 1, 256).to(self.device)
        self.network.update({'Fuse_sft_block': self.fuse_sft_block})

    def render_video(self, images, id_name, *args, **kwargs):
        """images -> video of geometry and depth, 3D consistency check.

        Args:
            all_rgb (torch.Tensor): input images
            save_gif (bool, optional): save as gif? Defaults to False.
        """
        import skvideo.io
        from tqdm import tqdm
        import torch
        from project.utils.misc_utils import Tensor2Array

        torch.cuda.empty_cache()
        opt = self.opt.inference

        # get input
        images = self.pool_256(images).to(self.device)

        # encode ref view
        ref_imgs_info = self.encode_ref_images(images)
        # ref_render_out = ref_imgs_info['global_render_out']
        pred_latents = ref_imgs_info['pred_latents']
        ref_view_aligned_feat = ref_imgs_info['ref_view_aligned_feat']  # !

        # generate ellipsoid trajectory
        chunk = 1
        trajectory = self.create_trajectory(self.opt.inference.video_frames)
        suffix = '_azim' if opt.azim_video else '_elipsoid'
        video_dst_dir = os.path.join(self.results_dst_dir, self.mode, 'videos',
                                     str(id_name))
        Path(video_dst_dir).mkdir(parents=True, exist_ok=True)

        # create writers
        video_filename = 'sample_video_{}.mp4'.format(suffix)
        writer = skvideo.io.FFmpegWriter(os.path.join(video_dst_dir,
                                                      video_filename),
                                         outputdict={
                                             '-pix_fmt': 'yuv420p',
                                             '-crf': '17'
                                         })
        if not opt.no_surface_renderings:
            depth_video_filename = 'sample_depth_video_{}.mp4'.format(suffix)
            depth_writer = skvideo.io.FFmpegWriter(os.path.join(
                video_dst_dir, depth_video_filename),
                                                   outputdict={
                                                       '-pix_fmt': 'yuv420p',
                                                       '-crf': '18',
                                                   })

        for j in tqdm(range(0, trajectory.shape[0], chunk)):

            if not j % self.opt.inference.video_interval == 0:
                continue  # for pick imgs

            torch.cuda.empty_cache()

            chunk_trajectory = trajectory[j:j +
                                          chunk]  # currently only 1 supported
            chunk_cam_settings = self._cam_locations_2_cam_settings(
                1, chunk_trajectory)

            que_info = self.latent2image(
                pred_latents,
                chunk_cam_settings,
                geometry_sample=None,  # just study pixel sup
                local_condition=None,
                sample_with_renderer=True,
                input_is_latent=True)  # add later

            # * cross recosntruction
            que_render_ref_out = self.que_render_given_ref(
                ref_imgs_info,
                que_info,
                feat_key='que_view',
                im_feat_for_query=ref_view_aligned_feat
            )  # cross reconstruction

            res_render_out = que_render_ref_out['res_render_out']
            output_frame = res_render_out['gen_imgs']

            utils.save_image(
                output_frame,
                Path(video_dst_dir) / f'{j}.png',
                nrow=1,
                normalize=True,
                value_range=(-1, 1),
            )

            utils.save_image(
                que_render_ref_out['que_img_for_vis'],
                # Path(video_dst_dir) / f'{j}images_for_vis.png',
                Path(video_dst_dir) / f'{j}_images_for_vis.jpg',
                nrow=1,
                normalize=True,
                value_range=(-1, 1),
            )

            # output_frame = que_render_ref_out['que_img_for_vis'] # * uncomment this to save intermediate results. suitable for debugging.

            writer.writeFrame(Tensor2Array(output_frame))

            # depth
            if not opt.no_surface_renderings:
                # st()
                mesh_image = self.render_depth_mesh(
                    pred_latents,
                    True,
                    chunk_trajectory.squeeze(),
                    cam_settings=chunk_cam_settings)[0]
                depth_writer.writeFrame(mesh_image[:])

                bgr_depth_image = mmcv.rgb2bgr(mesh_image)
                mmcv.imwrite(bgr_depth_image,
                             str(Path(video_dst_dir) / f'{j}_depth.png', ))

        writer.close()
        if not opt.no_surface_renderings:
            depth_writer.close()

        print(f'output dir: {os.path.join(video_dst_dir, video_filename)}')

    def _cycle_train_step(self, viewpair_noise=None, inference=False):
        opt = self.opt.training
        requires_grad(self.discriminator, False)

        # 1: get id-view-pair data
        if viewpair_noise is None:
            viewpair_noise = make_pair_same_noise(opt.synthetic_batch_size,
                                                  opt.style_dim, self.device)
        with torch.no_grad():
            synthetic_data_sample = self.synthetic_data_sample(viewpair_noise)

            curr_fake_imgs, random_3d_sample_batch, rand_cam_settings = (
                synthetic_data_sample[k]
                for k in ('fake_imgs', 'sample_batch', 'cam_settings'))
            curr_fake_imgs = self.pool_256(curr_fake_imgs) # view order: A B

        # curr_fake_imgs_for_log = _swap_odd_even_index_view(
        #     curr_fake_imgs)  # swap for loss, B A

        # 2: inference w codes, global
        ref_imgs_info = self.encode_ref_images(curr_fake_imgs,
                                               rand_cam_settings)
        ref_render_out = ref_imgs_info['global_render_out']

        # * create swap query dicts
        # since we use the pair sampling, code does not need to be swapped.
        que_info = self._swap_ref_render_out_to_que(ref_render_out)

        # * cross recosntruction
        que_render_ref_out = self.que_render_given_ref(
            ref_imgs_info, que_info)  # cross reconstruction
        # res_render_out['depth_mean'] = res_render_out['prj_mean'] # logistics distribution mean = depth here

        # * swap back for logging and reconstruction loss
        res_render_out = que_render_ref_out['res_render_out']
        res_gen_imgs = _swap_odd_even_index_view(
            res_render_out['gen_imgs'])  # swap for loss
        res_gen_thumb_imgs = _swap_odd_even_index_view(
            res_render_out['gen_thumb_imgs'])

        que_view_res_gt = _swap_odd_even_index_view(
            ref_imgs_info['res_gt']
        )  # ! fix bug, now just doing self reconstruction
        que_view_res_ada = _swap_odd_even_index_view(que_render_ref_out['aligned_res']) # A B order 

        # ? didn't swap the Reconstruciton for GT. stupid.
        res_render_out['gen_imgs'] = res_gen_imgs
        res_render_out['gen_thumb_imgs'] = res_gen_thumb_imgs

        # * details still missing
        res_gen_imgs_256 = self.pool_256(res_gen_imgs)
        res_gen_thumb_imgs_256 = torch.nn.functional.interpolate(
            res_gen_thumb_imgs, (256, 256))
        # gen_imgs_res = (curr_fake_imgs - res_gen_imgs_256
        #                 )  # already swapped back to A B order
        # res_restored = (curr_fake_imgs - res_gen_thumb_imgs_256
        #                 )  # res of I_recG and gt
        # ? has grad?
        # delta_thumb = res_restored - ref_imgs_info['res_gt']
        # delta = que_render_ref_out['aligned_res'] - ref_imgs_info['res_gt']

        # input - I_g - I_resgt - I^prime_g - I^prime_res - I^prime_highres

        images_for_vis = torch.cat([ # all following A B order 
            curr_fake_imgs, # swapped GT
            self.pool_256(ref_render_out['gen_thumb_imgs']),
            res_gen_thumb_imgs_256, # G0 reconstruction
            que_view_res_ada, # ada res
            ref_imgs_info['res_gt'], # res GT
            res_gen_imgs_256, # G1 cross creconstruction 
        ],
                                   dim=-1)  # concat in w dim.

        # rendering loss
        loss_dicts = self._compute_loss(curr_fake_imgs,
                                        res_render_out,
                                        geometry_sample=None)

        # * calculate residual align loss
        if self.opt.training.res_lambda > 0:
            loss_res = F.l1_loss(que_render_ref_out['aligned_res'],
                                 que_view_res_gt)
            #  ref_imgs_info['res_gt'])
            loss_dicts['loss'] += self.opt.training.res_lambda * loss_res
            loss_dicts['loss_dict'].update(dict(loss_res=loss_res))

        # if self.opt.training.res_lambda_thumb > 0:  # basically thumb img reconstruction loss
        #     loss_res_thumb = F.l1_loss(delta_thumb,
        #                                torch.zeros_like(delta_thumb))
        #     loss_dicts[
        #         'loss'] += self.opt.training.res_lambda_thumb * loss_res_thumb
        #     loss_dicts['loss_dict'].update(dict(loss_res_thumb=loss_res_thumb))

        # backward and upate
        if self.mode == 'train':
            loss_dicts['loss'].backward()

            self.optimizer_e.step()
            self.encoder.zero_grad()

        ret_dict = dict(
            loss_dict=loss_dicts['loss_dict'],
            render_out=res_render_out,
            rand_cam_settings=rand_cam_settings,
            images_for_vis=images_for_vis,
            viewpair_noise=viewpair_noise,
            ref_imgs_info=ref_imgs_info,
            pred_latents=ref_imgs_info['pred_latents'],
        )

        return ret_dict
