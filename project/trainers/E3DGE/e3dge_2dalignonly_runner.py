import os
from tqdm import tqdm
from pathlib import Path
from pdb import set_trace as st

from copy import deepcopy
import mmcv
import skvideo.io
import torch
import torch.nn.functional as F
from project.trainers.base_runner import RUNNER
# from project.trainers.cycle_trainer import CycleTrainer
from project.utils import requires_grad
from project.utils.training_utils import make_pair_same_noise, _swap_odd_even_index_view
from project.trainers.cycle_runner import CycleRunner
from project.utils.misc_utils import (PosEncoding, Tensor2Array)

from project.utils.dist_utils import get_rank, reduce_loss_dict
from torchvision import utils

from project.models.helper_modules.alignment_old import ResidualAligner
from project.utils.training_utils import make_pair_same_noise, _swap_odd_even_index_view
from project.utils.dist_utils import get_rank, reduce_loss_dict
from torchvision import utils


@RUNNER.register_module(force=True)
class E3DGE_2DAlignOnly_Runner(
        CycleRunner, ):
    """
    add PEinto projected local features, 3D Inpainter.
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
        super().__init__(
            encoder,
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
            #  optimizer,
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

        # * merged from cycle_trainer
        self.enable_swap_code = self.opt.training.swap_code
        self.swap_code = self.enable_swap_code

        self.enable_swap_res = self.opt.training.swap_res
        self.swap_res = self.enable_swap_res
        if self.swap_res:
            assert self.opt.training.enable_local_model, 'check flags'

        assert self.opt.training.enable_local_model

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

        ref_imgs_info = dict(
            **self.encoder_ref_images_global(images, cam_settings, no_editing,
                                             editing_boundary_scale_list,
                                             encoder_out), )

        return ref_imgs_info

    def que_render_given_ref(
            self,
            ref_imgs_info: dict,
            que_info: dict,
            self_rec=False,
            im_feat_for_query=None,
            #  edit_render_out=None,
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

        que_cam_settings = que_info['cam_settings']
        que_calibs = que_info['cam_settings']['calibs']

        assert que_wd_pts.ndim == 5  # B H W S 3
        B, H, W, S, _ = que_wd_pts.shape

        # reshape for query feat
        que_wd_pts_B3N = que_wd_pts.clone().reshape(B, -1, 3).permute(
            0, 2, 1)  # B 3 N, for feat query

        # * align with query view img + depth
        if 'depth' in self.opt.rendering.residual_context_feats:
            depth_as_feat = que_info['depth'].permute(0, 3, 4, 1,
                                                      2).squeeze(1)  # B 1 H W

            if res_gt.shape != depth_as_feat.shape:
                depth_as_feat = F.interpolate(depth_as_feat,
                                              size=res_gt.shape[-2:])
        else:
            depth_as_feat = None

        # 2.2 get the E feature map

        que_upsampled_thumb_imgs = torch.nn.functional.interpolate(
            que_info['gen_thumb_imgs'], (256, 256))
        aligned_res = self.grid_align(
            torch.cat((res_gt, que_upsampled_thumb_imgs),
                      dim=1))  # use the high-res

        # ! 3. calculate ref view residual feature map
        self.g_module.renderer.network.netLocal.filter(
            aligned_res,
            depth_as_feat,
            ref_feats=None,
            feat_key='que_view',
        )

        # create feature volumes
        # Get residual feats
        que_proj_ref_feats_BNC = self.g_module.renderer.network.netLocal.query(
            points=que_wd_pts_B3N,
            feat_key='que_view',
            calibs=que_calibs,  # !
            return_eikonal=False,
            return_feat_only=True,
            im_feat=im_feat_for_query)  # B, 256, N
        in_img_mask = que_proj_ref_feats_BNC['in_img'].reshape(B, H, W, S, 1)

        pix_aligned_residual_feats = que_proj_ref_feats_BNC['feats'].permute(
            0, 2, 1).reshape(B, H, W, S, -1)  # reshape from BCN

        # ============ add PE of que points ============== #
        residual_ctx_pe = self.PE(que_wd_pts)
        residual_ctx_pe = residual_ctx_pe.reshape(B, H, W, S, -1)

        pix_aligned_residual_feats = torch.cat(
            (pix_aligned_residual_feats, residual_ctx_pe), -1)  # 256+45

        # 5. high res reconstruct
        visible_local_condition = {
            'feats': pix_aligned_residual_feats,
        }  # image as input

        res_render_out = self.latent2image(
            ref_imgs_info['pred_latents'],
            que_cam_settings,
            local_condition=visible_local_condition)  # add later
        # ================ do 2D feature inpaint + 2D-3D hybrid feature alignment ==========#

        que_img_for_vis = torch.cat(
            (que_upsampled_thumb_imgs, self.pool_256(res_gt),
             self.pool_256(aligned_res),
             self.pool_256(res_render_out['gen_thumb_imgs']),
             self.pool_256(res_render_out['gen_imgs'])), -1)

        return dict(res_render_out=res_render_out, # B A order
                    aligned_res=aligned_res,
                    res_gt=res_gt,
                    que_img_for_vis=que_img_for_vis,
                    in_img_mask=in_img_mask)

    def encoder_ref_images_global(
        self,
        images,
        cam_settings=None,
        no_editing=False,
        editing_boundary_scale_list=None,
        encoder_out=None,
    ):
        input_thumb_imgs = self.pool_64(images)
        input_imgs = self.pool_256(images)

        # global part encoding
        if encoder_out is None:  # could be reused when rendering video
            encoder_out = super().image2latents(input_imgs,
                                                return_featmap=True)
        pred_latents, feat_maps = (encoder_out[k]
                                   for k in ('pred_latents', 'p32'))

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
        orig_res_gt = (input_imgs - render_out['gen_thumb_imgs']).detach()

        # 2. get context informaion
        if (self.opt.inference.editing_inference and not no_editing):
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
                torch.cat((orig_res_gt, edit_upsampled_thumb_imgs),
                          dim=1))  # use the high-res
            depth = edit_render_out['depth']  # B H W 1 1

        else:
            res_gt = orig_res_gt
            edit_render_out = None
            depth = render_out['depth']  # B H W 1 1
            que_latents = pred_latents

        # global part encoding results
        return dict(
            img_feats=feat_maps,
            orig_res_gt=orig_res_gt,
            cam_settings=pred_cam_settings,
            global_render_out=render_out,
            res_gt=res_gt,
            edit_render_out=edit_render_out,
            pred_latents=que_latents,
            depth=depth,
            imgs=input_imgs,
            encoder_out=encoder_out,
        )

    def image2image(self,
                    images: torch.Tensor,
                    cam_settings=None,
                    res_gt=None,
                    **kwargs):
        """encode reference images, reconstruct itself.

        Args:
            images (torch.Tensor): _description_
            ref_imgs_info (dict, optional): _description_. Defaults to None.
        """

        input_thumb_imgs = self.pool_64(images)
        if self.opt.training.full_pipeline:
            input_imgs = self.pool_256(images)
        else:
            input_imgs = input_thumb_imgs

        ref_imgs_info = self.encode_ref_images(input_imgs)
        ref_global_render_out = ref_imgs_info['global_render_out']

        with torch.no_grad():
            images_for_vis = torch.cat([
                self.pool_256(ref_imgs_info['imgs'].detach()),
                self.pool_256(
                    ref_global_render_out['gen_thumb_imgs'].detach()),
                self.pool_256(ref_imgs_info['res_gt'].detach())
            ],
                                       dim=-1)  # concat in w dim.

        # 2. self residual reconstruction
        que_render_out = self.que_render_given_ref(
            ref_imgs_info, ref_imgs_info['global_render_out'])
        res_render_out = que_render_out['res_render_out']

        with torch.no_grad():
            images_for_vis = torch.cat(
                [images_for_vis, que_render_out['que_img_for_vis']],
                dim=-1)  # concat in w dim.

        # match the original return api
        res_render_out.update(
            images_for_vis=images_for_vis,
            pred_latents=ref_imgs_info['pred_latents'],
            input_imgs=ref_imgs_info['imgs'].detach(),
            pred_cam_settings=ref_imgs_info['cam_settings'],
            upsampled_thumb_imgs=ref_global_render_out['gen_thumb_imgs'],
            que_render_out=que_render_out,
            res_gt=ref_imgs_info['res_gt'])
        return res_render_out

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
            curr_fake_imgs = self.pool_256(curr_fake_imgs)

        # 2: inference w codes, global
        ref_imgs_info = self.encode_ref_images(curr_fake_imgs,
                                               rand_cam_settings)
        ref_render_out = ref_imgs_info['global_render_out']

        # * create swap query dicts
        # since we use the pair sampling, code does not need to be swapped.
        que_info = self._swap_ref_render_out_to_que(ref_render_out)

        # * cross reconstruction
        que_render_ref_out = self.que_render_given_ref(
            ref_imgs_info, que_info)  # cross reconstruction

        # * swap back for logging and reconstruction loss
        res_render_out = que_render_ref_out['res_render_out']

        # ! debugging here, remove ada for debugging
        # que_render_ref_out['aligned_res'] = ref_imgs_info['res_gt']
        aligned_res = _swap_odd_even_index_view(
            que_render_ref_out['aligned_res'])  # swap for loss

        # aligned_res = _swap_odd_even_index_view(
        #     ref_imgs_info['res_gt'])  # swap for loss

        res_gen_imgs = _swap_odd_even_index_view(
            res_render_out['gen_imgs'])  # swap for loss
        res_gen_thumb_imgs = _swap_odd_even_index_view(
            res_render_out['gen_thumb_imgs'])

        que_view_res_gt = _swap_odd_even_index_view(
            ref_imgs_info['res_gt']
        )  # ! fix bug, now just doing self reconstruction

        curr_fake_imgs_for_log = _swap_odd_even_index_view(
            curr_fake_imgs)  # swap for loss

        # ? didn't swap the Reconstruciton for GT. stupid.
        res_render_out['gen_imgs'] = res_gen_imgs
        res_render_out['gen_thumb_imgs'] = res_gen_thumb_imgs

        # * details still missing
        res_gen_thumb_imgs_256 = torch.nn.functional.interpolate(
            res_gen_thumb_imgs, (256, 256))

        res_restored = (curr_fake_imgs - res_gen_thumb_imgs_256
                        )  # res of I_recG and gt
        # ? has grad?
        delta_thumb = res_restored - ref_imgs_info['res_gt']

        # input - I_g - I_resgt - I^prime_g - I^prime_res - I^prime_highres
        # rendering loss
        loss_dicts = self._compute_loss(curr_fake_imgs,
                                        res_render_out,
                                        geometry_sample=None,
                                        aligned_res=aligned_res)

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

        with torch.no_grad():
            images_for_vis = torch.cat([
                curr_fake_imgs_for_log,
                que_render_ref_out['que_img_for_vis'],
                # que_view_res_gt,
            ],
                                       dim=-1)  # concat in w dim.

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

    def train_step(self):
        self.train_mode()
        opt = self.opt.training
        iter_idx = self._iter
        loss_dicts_to_reduce = {}

        # == train D ===
        if self.train_discriminator and (
            (iter_idx % opt.train_discriminator_step_interval) == 0):
            d_loss_dicts = super()._train_discriminator_step()
            loss_dicts_to_reduce.update(d_loss_dicts)

        # == train E with cycle ===
        assert opt.cycle_training

        if self.train_synthetic:
            train_step_out = self._cycle_train_step()
        elif self.train_real:
            train_step_out = self.realimg_forward()

        loss_dicts_to_reduce.update(train_step_out['loss_dict'])
        images_for_vis = train_step_out['images_for_vis']

        loss_reduced = reduce_loss_dict(loss_dicts_to_reduce)

        if get_rank() == 0:
            self.wandb_log(loss_reduced)  # * save results

            if self.opt.training.progressive_pose_sampling and self.train_synthetic:
                self.wandb_log(
                    dict(azim_range=train_step_out['rand_cam_settings']
                         ['azim_range'],
                         elev_range=train_step_out['rand_cam_settings']
                         ['elev_range']))

            if iter_idx % self.opt.training.saveimg_interval == 0:
                utils.save_image(
                    images_for_vis[0:4],
                    os.path.join(
                        self.results_dst_dir,
                        self.mode,
                        #  f"images/{str(iter_idx).zfill(7)}.png"),
                        f"images/{str(iter_idx).zfill(7)}.jpg"),
                    nrow=images_for_vis[0:4].shape[-1] // 256,
                    normalize=True,
                    value_range=(-1, 1),
                )

        del loss_reduced
        del train_step_out

    def _swap_ref_render_out_to_que(self,
                                    ref_render_out,
                                    extra_keys: list = []):
        keys = [
            'xyz',
            'points',
            'dists',
            'cam_settings',
            # 'hit_prob',
            # 'rays_o',
            'gen_thumb_imgs',
            'depth'
        ] + extra_keys

        que_info = {}  # do information swapping.
        for k in keys:
            v = ref_render_out[k]

            if isinstance(v, torch.Tensor):  # avoid clone() fail for 'xyz'
                que_v = v.clone()
            else:
                que_v = deepcopy(v)  # for dict or modules

            if isinstance(que_v, dict):  # cam_settings
                for k_cam, v_cam in que_v.items():
                    if isinstance(v_cam, torch.Tensor):
                        que_v[k_cam] = _swap_odd_even_index_view(v_cam)
            elif isinstance(que_v, torch.Tensor):
                que_v = _swap_odd_even_index_view(que_v)

            elif isinstance(que_v, list):
                que_v = [
                    _swap_odd_even_index_view(que_v_item)
                    for que_v_item in que_v
                ]
            else:
                raise TypeError()

            que_info[k] = que_v

        return que_info

    # @staticmethod
    def _build_model(self):
        super()._build_model()
        if self.opt.training.enable_local_model:
            self.grid_align = ResidualAligner(self.opt.training).to(
                self.device)  #ADA
            self.network.update({'grid_align': self.grid_align})

        else:
            self.grid_align = None
        self.residue = None

    def _grad_flags(self):
        super()._grad_flags()
        opt_training = self.opt.training

        if self.opt.rendering.enable_local_model:
            requires_grad(self.g_module.renderer.network.netLocal,
                          not opt_training.E_l_grad_false)  # L

    def _get_trainable_parmas(self):
        opt = self.opt.training
        parent_params = super()._get_trainable_parmas()

        for k, v in self.network.items():
            if k in ['encoder', 'pifu']:
                continue  # already handled these two classes in trainer.py

            # if k in self.opt.training.ckpt_to_ignore:
            #     continue
        
            if k == 'grid_align' and opt.fix_ada:
                continue

            params_group = {
                'name': k,
                'params': v.parameters(),
                'lr': opt.ada_lr
            }
            parent_params.extend([params_group])

        return parent_params

    def render_video(self, images, id_name, *args, **kwargs):
        """images -> video of geometry and depth, 3D consistency check.

        Args:
            all_rgb (torch.Tensor): input images
            save_gif (bool, optional): save as gif? Defaults to False.
        """

        torch.cuda.empty_cache()
        opt = self.opt.inference

        # get input
        images = self.pool_256(images).to(self.device)

        # encode ref view
        ref_imgs_info = self.encode_ref_images(images)
        pred_latents = ref_imgs_info['pred_latents']

        # generate ellipsoid trajectory
        chunk = 1
        trajectory = self.create_trajectory(self.opt.inference.video_frames)
        suffix = '_azim' if opt.azim_video else '_elipsoid'
        video_dst_dir = os.path.join(self.results_dst_dir, self.mode, 'videos',
                                     str(id_name))
        Path(video_dst_dir).mkdir(parents=True, exist_ok=True)

        # create writers
        video_filename = 'sample_video_{}.mp4'.format(suffix)
        writer = skvideo.io.FFmpegWriter(
            os.path.join(video_dst_dir, video_filename),
            outputdict={
                '-pix_fmt': 'yuv420p',
                '-crf': '18',
                #  '-crf': '0'
            })
        if not opt.no_surface_renderings:
            depth_video_filename = 'sample_depth_video_{}.mp4'.format(suffix)

            if Path(os.path.join(video_dst_dir,
                                 depth_video_filename)).exists():
                print("ignore {}".format(
                    os.path.join(video_dst_dir, depth_video_filename)))
                return

            depth_writer = skvideo.io.FFmpegWriter(
                os.path.join(video_dst_dir, depth_video_filename),
                outputdict={
                    '-pix_fmt': 'yuv420p',
                    #  '-crf': '18',
                    '-crf': '1'
                })

        # * for loop render RGB and Depth. Start from Depth first.
        for j in tqdm(range(0, trajectory.shape[0], chunk)):
            torch.cuda.empty_cache()

            if not j % self.opt.inference.video_interval == 0:
                continue  # for pick imgs

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
                ref_imgs_info, que_info, True)  # cross reconstruction

            res_render_out = que_render_ref_out['res_render_out']
            output_frame = self.pool_256(res_render_out['gen_imgs'])

            # output_frame = que_render_ref_out['que_img_for_vis'] # * uncomment this to save intermediate results. suitable for debugging.
            writer.writeFrame(Tensor2Array(output_frame))

            # depth
            if not opt.no_surface_renderings:
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
