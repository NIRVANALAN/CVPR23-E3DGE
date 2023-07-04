import gc
import json
import os
from pathlib import Path

from pdb import set_trace as st
import cv2
import mmcv  # to replace cv2 and plt
import numpy as np
import skvideo.io
import torch
import torch.nn as nn
import torch.nn.functional as F
from facexlib.alignment import init_alignment_model
from facexlib.visualization import visualize_alignment
from munch import Munch
from pytorch3d.structures import Meshes
from torch.utils.data import DataLoader
from torchvision import utils
from torchvision.transforms import GaussianBlur
from tqdm import tqdm
import traceback

import project.utils.deca_util as util
from project.data.dataset import ImagesDatasetEval
from project.data.now import NoWDataset
from project.losses.gan_loss import (d_logistic_loss, d_r1_loss, eikonal_loss,
                                     g_nonsaturating_loss, viewpoints_loss,
                                     calculate_adaptive_weight)
from project.utils import (Ranger, add_textures, create_cameras,
                           create_mesh_renderer, generate_camera_params,
                           gt_pool, mixing_noise, print_parameter,
                           refresh_cache, requires_grad, sample_data, xyz2mesh)
from project.utils.dist_utils import get_rank, reduce_loss_dict
from project.utils.mesh_utils import (align_volume,
                                      extract_mesh_with_marching_cubes)
from project.utils.misc_utils import (Tensor2Array, crop_one_img,
                                      landmark_98_to_7, plt_2_cv2, vis_tensor,
                                      get_trainable_parameter)

from .base_runner import RUNNER
from .datasetgan_runner import DatasetGANRunner

blur = GaussianBlur(kernel_size=(9, 9), sigma=(0.1, 2.0))

try:
    import wandb
except ImportError:
    wandb = None


@RUNNER.register_module(force=True)
class AERunner(DatasetGANRunner):
    """A naive re-implmenetation of IterBasedRunner building over BaseRunner

    Args:
        BaseRunner (_type_): _description_
    """

    def __init__(
        self,
        encoder: torch.nn.Module,
        volume_discriminator: torch.nn.Module,
        generator: torch.nn.Module,
        mean_latent,
        opt: Munch,
        device: torch.device,
        loaders: dict,
        loss_class: torch.nn.Module,
        surface_g_ema: torch.nn.Module,
        ckpt=None,  # todo, to lint all params
        mode='val',  # todo
        work_dir=None,
        max_iters=None,
        max_epochs=None,
        discriminator=None,
    ):

        super().__init__(encoder, volume_discriminator, generator, mean_latent,
                         opt, device, loaders, loss_class, surface_g_ema, ckpt,
                         mode, work_dir, max_iters, max_epochs, discriminator)
        self.volume_discriminator = volume_discriminator
        self.opt = opt
        self.opt_experiment = opt.experiment
        self.generator = generator
        self.discriminator = discriminator
        self.device = device
        self.mean_latent = mean_latent
        self.w_mean_latent = [
            latent[:, 0, :] if latent is not None else latent
            for latent in mean_latent
        ]

        self.loss_class = loss_class

        # creating datasets
        self.train_loader = sample_data(loaders['train'])
        self.eval_loader, self.test_loader = (loaders[k]
                                              for k in ('val', 'test'))
        self.loaders = loaders

        # fixed camera setting for evaluation
        self.locations = torch.tensor(
            [[0, 0], [-1.5 * opt.camera.azim, 0], [-1 * opt.camera.azim, 0],
             [-0.5 * opt.camera.azim, 0], [0.5 * opt.camera.azim, 0],
             [1 * opt.camera.azim, 0], [1.5 * opt.camera.azim, 0],
             [0, -1.5 * opt.camera.elev], [0, -1 * opt.camera.elev],
             [0, -0.5 * opt.camera.elev], [0, 0.5 * opt.camera.elev],
             [0, 1 * opt.camera.elev], [0, 1.5 * opt.camera.elev]],
            device=self.device)
        self.frontal_location = torch.tensor([[0.0, 0.0]]).to(self.device)

        self.frontal_cam_settings = self._cam_locations_2_cam_settings(
            batch=1, cam_locations=self.frontal_location)

        # todo, extra network weights here.
        self.network = {}
        self._build_model()
        self._load_ckpt(ckpt)
        self._post_load_ckpt(ckpt)
        self._dist_prepare()

        if mode == 'train':
            assert loss_class is not None, 'send in loss_module here'
            self.setup_optimizers()
            self.train_mode()
        else:
            self.eval_mode()

        # * shared flags
        self.opt.training.enable_G1 = self.opt.training.enable_G1
        self.results_dst_dir = self.opt.results_dst_dir
        self.reset_train_flag()

        self.load_editing_dir = False
        if self.opt.inference.editing_inference:
            self._load_editing_directions()

        print('trainer init done!, device: {}'.format(self.device))

    def reset_train_flag(self):
        self.train_real, self.train_synthetic = False, False

    def run(self):
        # * save ckpt + roading between different train_step() + run validation_step()
        self.on_train_step_start(self.train_loader)
        train_opt = self.opt.training
        torch.set_grad_enabled(True)

        for iter_idx in self._pbar:
            self._iter = iter_idx + train_opt.start_iter
            self.zero_grad()
            self.train_mode()

            if not train_opt.evaluate_in_train and train_opt.synthetic_sampling_strategy == 'all_fake' or (
                    train_opt.synthetic_sampling_strategy == 'hybrid'
                    and self._iter % 2 == 1):
                self.train_synthetic = True
            if not train_opt.evaluate_in_train and train_opt.synthetic_sampling_strategy == 'all_real' or (
                    train_opt.synthetic_sampling_strategy == 'hybrid'
                    and self._iter % 2 == 0):
                self.train_real = True

            self.train_step()  # hard coded

            iter_trained_over = 'synthetic' if self.synthetic_data_flag else 'real'

            refresh_cache()

            if get_rank() == 0:
                description = f"Iter: {self._iter} SampledFrom: {iter_trained_over:10} GPU: {os.environ['CUDA_VISIBLE_DEVICES']}"
                self._pbar.set_description((description))
                # try:
                if iter_idx % self.opt.training.val_interval == 0 and not self.opt.inference.no_eval:
                    self.validation(mode='val')
                    refresh_cache()

                if self._iter % self.opt.training.ckpt_interval == 0 and self.mode == 'train':
                    self.save_network(filename_tmpl=f'{iter_idx:07}')

                if self._iter % 1000 == 0:
                    self.save_network(filename_tmpl=f'latest')

        self.on_train_step_end()

    # todo, rename or be merged?
    def forward_step(self):
        forward_out = {}

        if self.train_synthetic:
            synthetic_out = self.synthetic_forward()
            forward_out = synthetic_out

        if self.train_real:
            real_out = self.realimg_forward()
            forward_out = real_out

        return forward_out

    @torch.enable_grad()
    def train_step(self, _forward_callback=None):
        """run all _step_callback and logging

        Args:
            _step_callback (function, optional): forward functin to call. Defaults to None.

        Returns:
            tuple: description_log, loss_dict
        """
        self.train_mode()
        if _forward_callback is None:
            _forward_callback = self.forward_step

        loss_dicts_to_reduce = {}

        forward_callback_out = _forward_callback()
        assert isinstance(forward_callback_out,
                          dict) and 'loss_dict' in forward_callback_out
        loss_dicts_to_reduce.update(
            forward_callback_out['loss_dict'])  # type: ignore
        loss_reduced = reduce_loss_dict(loss_dicts_to_reduce)

        # iter_trained_over = 'synthetic' if self.synthetic_data_flag else 'real'
        self.reset_train_flag()

        # * todo, merge other train_steps
        if get_rank() == 0:
            self.wandb_log(loss_reduced)

            if self._iter % self.opt.training.saveimg_interval == 0:
                images_for_vis = forward_callback_out['images_for_vis']
                utils.save_image(
                    images_for_vis[0:4],
                    os.path.join(self.results_dst_dir, self.mode,
                                 f"images/{str(self._iter).zfill(7)}.jpg"),
                    nrow=images_for_vis[0:4].shape[-1] // 256,
                    normalize=True,
                    value_range=(-1, 1),
                )

        del loss_reduced, forward_callback_out

    def wandb_log(self, loss_reduced):
        try:
            if wandb and self.opt.training.wandb:  # todo
                wandb_log_dict = {
                    k: loss_reduced[k].mean().item() if isinstance(
                        loss_reduced[k], torch.Tensor) else loss_reduced[k]
                    for k in loss_reduced.keys()
                }
                wandb.log(wandb_log_dict)
        except:
            traceback.print_exc()

    @torch.no_grad()
    def visualization(self):  # TODO
        # multi-view render img, depth
        # with volume render and mesh render
        # save video
        pass

    @torch.no_grad()
    def on_val_start(self, mode):
        torch.cuda.empty_cache()
        self.eval_mode()
        self.mode = mode
        loader = self.loaders[self.mode]

        metrics_root_dir = os.path.join(self.opt.results_dst_dir, self.mode,
                                        str(self._iter), 'images_for_metrics')
        metrics_root_dir = Path(metrics_root_dir)
        if self.opt.inference.save_independent_img:  # for 2D metrics
            metrics_root_dir.mkdir(exist_ok=True, parents=True)

        # analysis_imgsave_root_dir = Path(os.path.join(self.opt.results_dst_dir, self.mode, str(self._iter), 'validation_images'))
        val_imgsave_root_dir = Path(
            os.path.join(self.opt.results_dst_dir, self.mode, str(self._iter)))
        print('validation save dir: {}'.format(val_imgsave_root_dir))
        val_imgsave_root_dir.mkdir(exist_ok=True, parents=True)
        (val_imgsave_root_dir / 'images_for_vis').mkdir(exist_ok=True,
                                                        parents=True)
        (val_imgsave_root_dir / 'meshes').mkdir(exist_ok=True, parents=True)
        return loader, val_imgsave_root_dir, metrics_root_dir

    @torch.inference_mode()
    def on_val_end(self):
        torch.cuda.empty_cache()
        gc.collect()

    @torch.inference_mode()
    def validation(self, eval_imgs=4500, mode='val'):

        # TODO, include identities from cli

        all_loss_dicts = []
        opt = self.opt.inference
        loader, val_imgsave_root_dir, metrics_root_dir = self.on_val_start(
            mode)

        smile_ids = self.opt.inference.smile_ids
        age_ids = self.opt.inference.age_ids
        bangs_ids = self.opt.inference.bangs_ids
        beard_ids = self.opt.inference.beard_ids

        ids_edit = dict(
            Bangs=bangs_ids,
            Smiling=smile_ids,
            No_Beard=beard_ids,
            Young=age_ids,
        )

        id_attribute_map = dict()
        for attribute, ids in ids_edit.items():
            for id_no in ids:
                id_attribute_map[str(id_no)] = attribute

        identities_to_inference = smile_ids + bangs_ids + age_ids + beard_ids  # * as filter

        all_pred_latents = []

        # inference celebamask hq ids list
        if self.opt.inference.video_output_csv != '':
            with open(self.opt.inference.video_output_csv, 'r') as f:
                content = f.readlines()
                identities_to_inference = [
                    content[i].split(',')[0] for i in range(len(content))
                ]
        else:
            identities_to_inference = []
            print(identities_to_inference)

        for idx, batch in enumerate(tqdm(loader)):

            images_paths = batch['img_path']

            img_stem = Path(images_paths[0]).stem

            # * choose the right identities for inference
            if self.opt.inference.editing_inference:
                if img_stem not in id_attribute_map:
                    print(f'{img_stem} not in attribute map')
                    continue
                attr_to_edit = id_attribute_map[img_stem]
                attr_to_edit_idx = self.ATTRS.index(attr_to_edit)
            else:
                if len(identities_to_inference
                       ) != 0 and img_stem not in identities_to_inference:
                    continue

            images = batch['image'].to(self.device)

            real_thumb_imgs = self.pool_64(images)
            images = self.pool_256(images)

            if self.opt.projection.inference_projection_validation:
                # w_inv_root = Path(self.opt.projection.w_inversion_root)
                inversed_w_latent_in = Path(
                    self.opt.projection.w_inversion_root
                ) / img_stem / 'latent_in.pt'
                # w_inv_root = Path(self.opt.projection.w_inversion_root)
                inversed_w_latent_in = Path(
                    self.opt.projection.w_inversion_root
                ) / img_stem / 'latent_in.pt'
                # st()
                assert inversed_w_latent_in.exists()
                inversed_latent = torch.load(
                    inversed_w_latent_in,
                    map_location=lambda storage, loc: storage)
                latent_in = inversed_latent['latent_in']
                pred_latents = [latent.to(self.device) for latent in latent_in]

                if self.opt.projection.PTI:
                    pti_g_state = inversed_latent['g']
                    self.g_module.load_state_dict(pti_g_state,
                                                  True)  # type: ignore
                    self.surface_g_ema.load_state_dict(pti_g_state, False)
                    print('load PTI weights')

                pred_cam_settings = self.image2camsettings(real_thumb_imgs)

                render_out = self.latent2image(
                    pred_latents, pred_cam_settings)  # type: ignore

                images_for_vis = torch.cat([
                    self.pool_256(images),
                    self.pool_256(render_out['gen_thumb_imgs']).detach()
                ],
                                           dim=-1)  # concat in w dim.
                if self.opt.training.enable_G1:
                    images_for_vis = torch.cat([
                        images_for_vis,
                        self.pool_256(render_out['gen_imgs']).detach()
                    ],
                                               dim=-1)

                render_out.update(
                    dict(
                        input_thumb_imgs=real_thumb_imgs,
                        pred_latents=pred_latents,
                        input_imgs=images,
                        images_for_vis=images_for_vis,
                        pred_cam_settings=pred_cam_settings,
                        cam_settings=pred_cam_settings,
                    ))

            else:
                torch.cuda.empty_cache()
                render_out = self.image2image(images,
                                              cam_settings=batch.get(
                                                  'cam_settings', None))
                all_pred_latents.append(render_out['pred_latents'])

            if 'render_out' in render_out:  # compatible
                render_out = render_out['render_out']

            if self.opt.training.enable_G1:  # todo, merge into trainer
                pred_imgs = self.pool_256(render_out['gen_imgs'])
                gt_imgs = images
            else:
                pred_imgs = render_out['gen_thumb_imgs']
                gt_imgs = real_thumb_imgs

            _, loss_2d_rec_dict = self.loss_module.calc_2d_rec_loss(  # type: ignore
                pred_imgs,
                gt_imgs,
                gt_imgs,
                self.opt.training,
                loss_dict=True,
                mode='val')  # 21.9G

            #=================== log metric =================
            loss_dict = {**loss_2d_rec_dict}
            all_loss_dicts.append(loss_dict)

            pred_gt_samples = torch.cat(
                [self.pool_256(pred_imgs),
                 self.pool_256(gt_imgs)], -2)  # in H dim

            if self.opt.training.enable_G1:  # save both imgs
                thumb_pred_gt_samples = torch.cat([
                    self.pool_64(render_out['gen_thumb_imgs']),
                    self.pool_64(gt_imgs)
                ], -2)
            else:
                thumb_pred_gt_samples = None

            if self.opt.inference.editing_inference:
                edit_render_out = self.edit_images(images, render_out)
                assert isinstance(edit_render_out, dict)
                render_out['images_for_vis'] = torch.cat(
                    (render_out['images_for_vis'],
                     self.pool_256(edit_render_out['edit_gen_thumb_imgs']),
                     self.pool_256(edit_render_out['gen_imgs'])), -1)
                # if self.opt.editing.render_video_for_each_direction:
                    # for i in range(4):
                    #     self.render_edit_video(
                    #         images,
                    #         id_name=Path(images_paths[0]).stem + '/' +
                    #         self.ATTRS[i],
                    #         attr_idx=i,
                    #         encoder_out=render_out)
                # else:
                self.render_edit_video(images,
                                        id_name=Path(images_paths[0]).stem + '/' +
                                        self.ATTRS[attr_to_edit_idx],
                                        attr_idx=attr_to_edit_idx,
                                        encoder_out=render_out)

            if self.opt.inference.render_video:
                assert gt_imgs.shape[0] == 1
                self.render_video(images, id_name=Path(images_paths[0]).name)

            else:
                if self.opt.inference.render_video:
                    assert gt_imgs.shape[0] == 1
                    self.render_video(images,
                                      id_name=Path(images_paths[0]).name)

            # ====== visualizations ==== #
            if idx < eval_imgs:

                for i in range(pred_imgs.shape[0]):
                    img_name = Path(
                        images_paths[i]).with_suffix('.png').name  # jpg -> png

                    # save analysis images
                    if not opt.no_surface_renderings:
                        surface_out = self.latent2surface(
                            render_out['pred_latents'][i:i + 1],
                            True,
                            cam_settings=render_out['pred_cam_settings'],
                            return_mesh=True,
                            local_condition=None)  # type: ignore

                        # also save as depth
                        self.save_depth_mesh(
                            surface_out,
                            cam_settings=render_out['pred_cam_settings'],
                            mesh_saveprefix=img_name.split('.')[0],
                            mesh_saveroot=str(val_imgsave_root_dir / 'meshes'))

                    pred_img_save_path = val_imgsave_root_dir / 'images_for_vis' / img_name
                    if 'images_for_vis' in render_out:
                        utils.save_image(
                            render_out['images_for_vis'][i],
                            pred_img_save_path,
                            nrow=1,
                            normalize=True,
                            value_range=(-1, 1),
                        )

                    if self.opt.inference.save_independent_img:
                        pred_img_save_path = metrics_root_dir / img_name

                        utils.save_image(
                            pred_imgs[i],
                            pred_img_save_path,
                            nrow=1,
                            normalize=True,
                            value_range=(-1, 1),
                        )

                else:
                    if thumb_pred_gt_samples is not None:
                        utils.save_image(
                            thumb_pred_gt_samples,
                            os.path.join(str(val_imgsave_root_dir),
                                         f"{idx}_thumb_pred_gt.jpg"),
                            nrow=thumb_pred_gt_samples.shape[0],
                            normalize=True,
                            value_range=(-1, 1),
                        )

                    utils.save_image(
                        pred_gt_samples,
                        os.path.join(str(val_imgsave_root_dir),
                                     f"{idx}_pred_gt.jpg"),
                        nrow=pred_gt_samples.shape[0],
                        normalize=True,
                        value_range=(-1, 1),
                    )

            del render_out, images, gt_imgs, pred_imgs, pred_gt_samples, thumb_pred_gt_samples
            torch.cuda.empty_cache()
            gc.collect()

        torch.save({'pred_latents': all_pred_latents},
                   os.path.join(str(val_imgsave_root_dir), 'pred_latents.pt'))
        val_scores_for_logging = self._calc_average_loss(all_loss_dicts)

        # todo, add dataset header
        with open(os.path.join(str(val_imgsave_root_dir), 'scores.json'),
                  'a') as f:
            json.dump({**val_scores_for_logging, 'iter': self._iter}, f)

        self.on_val_end()

    def _calc_average_loss(self, all_loss_dicts):
        all_scores = {}  # todo, defaultdict
        mean_all_scores = {}

        for loss_dict in all_loss_dicts:
            for k, v in loss_dict.items():
                v = v.item()
                if k not in all_scores:
                    # all_scores[f'{k}_val'] = [v]
                    all_scores[k] = [v]
                else:
                    all_scores[k].append(v)

        for k, v in all_scores.items():
            mean = np.mean(v)
            std = np.std(v)
            if k in ['loss_lpis', 'loss_ssim']:
                mean = 1 - mean
            result_str = '{} average loss is {:.4f} +- {:.4f}'.format(
                k, mean, std)
            mean_all_scores[k] = mean
            print(result_str)

        val_scores_for_logging = {
            f'{k}_val': v
            for k, v in mean_all_scores.items()
        }
        return val_scores_for_logging

    def realimg_forward(self):
        opt = self.opt.training
        self.synthetic_data_flag = False
        self.zero_grad()

        if self.enable_adv_training:
            self._toggle_adversarial_loss('e_step')

        batch: dict = next(self.train_loader)
        # todo, add Decoder support later

        img, thumb_img = batch['img'], batch['thumb_img']
        img = self.pool_256(img)
        img = img.to(self.device)
        thumb_img = thumb_img.to(self.device)

        # rec
        render_out = self.image2image(img)

        images_for_vis = torch.cat(
            [
                img,
                self.pool_256(render_out['gen_thumb_imgs'])  # todo
            ],
            dim=-1)

        loss_dict = self._compute_loss(img,
                                       render_out,
                                       None,
                                       lms=batch.get('lms', None))

        if opt.eikonal_lambda > 0:
            g_eikonal, g_minimal_surface = eikonal_loss(
                render_out['eikonal_term'],
                sdf=render_out['sdf'] if opt.min_surf_lambda > 0 else None,
                beta=opt.min_surf_beta)
            g_eikonal = opt.eikonal_lambda * g_eikonal
            loss_dict.update({'real_eikonal_term': g_eikonal})
            if opt.min_surf_lambda > 0:
                g_minimal_surface = opt.min_surf_lambda * g_minimal_surface
        else:
            g_eikonal = torch.tensor(0, device=self.device)

        if opt.enable_G1:
            pred_imgs = self.pool_256(render_out['gen_imgs'])
        else:
            pred_imgs = self.pool_256(render_out['gen_thumb_imgs'])

        images_for_vis = torch.cat(
            [images_for_vis, self.pool_256(pred_imgs)], dim=-1)

        loss = (loss_dict['loss'] + g_eikonal) * self.opt.training.real_lambda
        loss.backward()

        self.optimizer_e.step()
        self.encoder.zero_grad()

        return dict(
            loss=loss,
            loss_dict=loss_dict['loss_dict'],
            render_out=render_out,
            batch=batch,
            images_for_vis=images_for_vis,
        )

    # todo: new class

    def synthetic_forward(self,
                          save_prefix='',
                          noise=None,
                          save_image=True,
                          compute_loss=True,
                          **kwargs):

        # todo, rename to _ste
        opt = self.opt.training
        self.zero_grad()

        # assert noise is not None
        if noise is None:
            noise = mixing_noise(opt.synthetic_batch_size, opt.style_dim,
                                 opt.mixing, self.device)  # * todo
        else:
            noise = [n.detach() for n in noise]

        for j in range(0, opt.synthetic_batch_size, opt.chunk):

            # get data
            synthetic_data_sample = self.synthetic_data_sample(
                noise[j:j + opt.chunk])
            curr_fake_imgs, random_3d_sample_batch, curr_rand_cam_settings = (
                synthetic_data_sample[k]
                for k in ('fake_imgs', 'sample_batch', 'cam_settings'))

            if not opt.pix_sup_only:
                geometry_sample = random_3d_sample_batch
            else:
                geometry_sample = None

            curr_render_out = self.image2image(curr_fake_imgs,
                                               curr_rand_cam_settings,
                                               geometry_sample=geometry_sample,
                                               **kwargs)

            images_for_vis = curr_render_out['images_for_vis']

            if opt.enable_G1:
                pred_imgs = self.pool_256(curr_render_out['gen_imgs'])
                images_for_vis = torch.cat(
                    [images_for_vis, pred_imgs
                     ],  # first cat itself, check done
                    -1)

            # loss & update grad
            ret_dict = dict(noise=noise[j:j + opt.chunk],
                            random_3d_sample_batch=random_3d_sample_batch,
                            render_out=curr_render_out,
                            images_for_vis=images_for_vis)

            curr_e_loss_dicts = self._compute_loss(curr_fake_imgs,
                                                   curr_render_out,
                                                   geometry_sample)
            # todo, merge to compute_loss
            if opt.latent_gt_lambda > 0:
                gt_latents = self.g_module.styles_and_noise_forward(  # type: ignore
                    noise[j:j + opt.chunk])
                # extend to w+
                gt_latents_wplus = [
                    latent.repeat(1, 9).reshape(-1, 9, 256)
                    for latent in gt_latents
                ]
                pred_latents = curr_render_out['pred_latents']

                loss_latent_rec = F.mse_loss(pred_latents[0],
                                             gt_latents_wplus[0])

                curr_e_loss_dicts['loss_dict'].update(
                    dict(loss_latent_gt=loss_latent_rec))
                curr_e_loss_dicts[
                    'loss'] += opt.latent_gt_lambda * loss_latent_rec

            curr_e_loss_dicts['loss'].backward()  # type: ignore

        self.optimizer_e.step()
        self.encoder.zero_grad()

        ret_dict.update(
            loss_dict=curr_e_loss_dicts['loss_dict'])  # type: ignore

        return ret_dict  # type: ignore

    @torch.no_grad()
    def log_images(self):
        # TODO, merge all log images here
        pass

    def on_train_step_start(self, loader):
        super().on_train_step_start(loader)
        opt_training = self.opt.training
        self.train_mode()
        batch = next(loader)  # for shape reference
        img, thumb_img = batch['img'], batch['thumb_img']
        self.img_shape = self.pool_256(img).shape
        self.thumb_shape = thumb_img.shape
        self.fixed_latents = None

        if get_rank() == 0:
            print('--- params in E0 ---')
            print_parameter(self.e_module)
            print('--- params in generator---')
            print_parameter(self.g_module)

        # if get_rank() == 0:
        # pbar = range(opt.iter)
        pbar = range(1000000)
        self._pbar = tqdm(pbar,
                          initial=opt_training.start_iter,
                          dynamic_ncols=True,
                          smoothing=0.01)

    def on_train_step_end(self):
        if get_rank() == 0:
            self.save_network('latest')
            print('Successfully saved final model.')

    # only global img by default
    def image2image(
            self,
            images,
            cam_settings=None,
            # local_condition=None,
            geometry_sample=None,
            render_surface=False):
        # mesh_savename=None):
        """atomic fn for 3dae

        Args:
            images (torch.Tensor): input images

        Returns:
            dict: dict with all generated images
        """
        # pred global latents
        torch.set_grad_enabled(
            self.mode ==
            'train')  # todo, move out of cycle trainer no grad mode
        if self.opt.training.full_pipeline:
            input_imgs = self.pool_256(images)
            input_thumb_imgs = self.pool_64(images)
        else:
            input_imgs = self.pool_64(images)
            input_thumb_imgs = input_imgs

        pred_latents = self.image2latents(input_imgs)

        # if cam_settings is None:
        pred_cam_settings = self.image2camsettings(input_thumb_imgs)

        render_out = self.latent2image(
            pred_latents,  # type: ignore
            pred_cam_settings,  # type: ignore
            geometry_sample=geometry_sample)
        if render_surface:
            surface_out = self.latent2surface(
                pred_latents,  # type: ignore
                True,
                cam_settings=cam_settings,  # type: ignore
                return_mesh=True)
            #   local_condition=local_condition)
            render_out.update(dict(surface_out=surface_out))

        images_for_vis = torch.cat([
            self.pool_256(input_imgs),
            self.pool_256(render_out['gen_thumb_imgs']).detach()
        ],
                                   dim=-1)  # concat in w dim.
        if self.opt.training.enable_G1:
            images_for_vis = torch.cat([
                images_for_vis,
                self.pool_256(render_out['gen_imgs']).detach()
            ],
                                       dim=-1)

        render_out.update(
            dict(input_thumb_imgs=input_thumb_imgs,
                 pred_latents=pred_latents,
                 input_imgs=input_imgs,
                 images_for_vis=images_for_vis,
                 cam_settings=cam_settings))

        if pred_cam_settings is not None:
            render_out.update(dict(pred_cam_settings=pred_cam_settings))

        return render_out

    # * done
    def latent2image(
            self,
            pred_latents: list,
            cam_settings: dict,
            local_condition=None,
            geometry_sample=None,  # for 3d sup
            sampling=False,
            input_is_latent=True,
            truncation=1,
            sample_with_renderer=False,
            **kwargs):
        # latents (global code + local feature_map) -> hat(images)
        # todo, input local_res not feature-map here
        opt = self.opt.training

        if local_condition is not None:
            assert opt.enable_local_model and not sampling

            if not isinstance(local_condition, dict):
                batch_size = pred_latents[0].shape[0]  #
                assert local_condition.shape[0] == batch_size
                local_data_batch = {
                    'gen_imgs': local_condition,
                    'calibs': cam_settings['calibs']
                }  # image as input
            else:
                local_data_batch = local_condition  # dict as input

        else:
            local_data_batch = None

        if 'pti_generator' in kwargs and kwargs[
                'pti_generator'] is not None:  # for PTI inversion
            generator = kwargs['pti_generator']
        else:
            generator = self.generator

        # render_out = self.generator(  # output size: <opt.chunk
        render_out = generator(  # output size: <opt.chunk
            pred_latents,  # all encoder should output same shape, regularize
            cam_settings['poses'],
            cam_settings['focal'],
            cam_settings['near'],
            cam_settings['far'],
            truncation=truncation,
            input_is_latent=input_is_latent,  # from synthetic samples
            geometry_sample=geometry_sample,
            truncation_latent=self.w_mean_latent,  # for synthetic sampling
            sample_with_renderer=not self.opt.training.enable_G1
            or sample_with_renderer,
            return_surface_eikonal=opt.return_surface_eikonal
            and self.mode == 'train',
            return_eikonal=self.enable_eikonal and self.mode == 'train',
            local_data_batch=local_data_batch,
            **kwargs)
        render_out.update(cam_settings=cam_settings, pred_latents=pred_latents)

        return render_out

    def latent2frontal_image(self, latents: list, input_is_latent, **kwargs):
        batch_size = latents[0].shape[0]

        # frontal_localtions = torch.tensor([[0, 0]])
        frontal_locations = self.frontal_location.repeat(batch_size,
                                                         1).to(self.device)

        frontal_cam_settings = self._cam_locations_2_cam_settings(
            batch=batch_size, cam_locations=frontal_locations)
        return self.latent2image(latents,
                                 frontal_cam_settings,
                                 input_is_latent=input_is_latent,
                                 sampling=True)

    @torch.no_grad()
    def generate_images(self, num_latents, direction='frontal'):
        latents = mixing_noise(num_latents, self.opt.training.style_dim,
                               self.opt.training.mixing, self.device)
        render_out: dict = self.latent2frontal_image(latents,
                                                     input_is_latent=False)

        gen_thumb_imgs = AERunner.tensor2numpy(render_out['gen_thumb_imgs'])
        gen_imgs = AERunner.tensor2numpy(render_out['gen_imgs'])

        return gen_thumb_imgs, gen_imgs, render_out

    @staticmethod
    @torch.no_grad()
    def tensor2numpy(tensor, normalize=True):
        if normalize:
            tensor = AERunner.normalize(tensor)
        return tensor.cpu().permute(0, 2, 3, 1).numpy()

    def image2camsettings(self, input_img, is_thumb=False):
        if not is_thumb:
            thumb_img = self.pool_64(input_img)
        else:
            thumb_img = input_img

        assert thumb_img.shape[
            -1] == 64, 'check volume_discriminator input img dim'

        with torch.no_grad():
            _, pred_locations = self.volume_discriminator(thumb_img)
        pred_cam_settings = self._cam_locations_2_cam_settings(
            batch=input_img.shape[0], cam_locations=pred_locations)
        return pred_cam_settings

    def image2latents(self, images, **kwargs):
        images = images.to(self.device)

        input_imgs = self.pool_256(images)
        # if self.opt.training.full_pipeline:
        #     input_imgs = self.pool_256(images)
        # else:
        #     input_imgs = self.pool_64(images)

        encoder_out = self.encoder(input_imgs, **kwargs)

        if isinstance(encoder_out, dict):
            pred_latents = encoder_out['pred_latents']
            encoder_out['pred_latents'] = self._add_offset2latent(
                pred_latents, self.device)
        else:
            pred_latents = encoder_out
            encoder_out = self._add_offset2latent(pred_latents, self.device)

        return encoder_out

    def _cam_locations_2_cam_settings(self, batch,
                                      cam_locations: torch.Tensor):
        opt = self.opt.training
        device = self.device
        cam_settings = generate_camera_params(
            opt.renderer_output_size,
            device,
            batch,
            locations=cam_locations,
            #input_fov=fov,
            uniform=opt.camera.uniform,
            azim_range=opt.camera.azim,
            elev_range=opt.camera.elev,
            fov_ang=opt.camera.fov,
            dist_radius=opt.camera.dist_radius,
            return_calibs=True)
        return cam_settings

    def _add_offset2latent(self, pred_offsets, device, w_space=False):
        """add wspace mean to pred_latents

        Args:
            pred_offsets (list): list of offsets
            mean_latent (list): mean_latent of two generator
            device (torch.device): cuda?
            w_space (bool, optional): use wspace truncation. Defaults to False.

        Returns:
            list: predicted latents in w-space
        """

        for latent_idx in range(2):  # todo, merge
            if pred_offsets[latent_idx] is not None:
                if w_space:
                    repeat_latent = self.mean_latent[latent_idx].repeat(
                        pred_offsets[0].shape[0], 1).to(device)  # B, 256
                else:
                    repeat_latent = self.mean_latent[latent_idx].repeat(
                        pred_offsets[0].shape[0], 1,
                        1).to(device)  # B 9 256, B 10 512
                    assert repeat_latent.shape == pred_offsets[
                        latent_idx].shape  # ndim=3, wp space
                    pred_offsets[
                        latent_idx] = repeat_latent + pred_offsets[latent_idx]
        return pred_offsets

    def _calculate_pixel_rec_loss(self,
                                  input_image: torch.Tensor,
                                  render_out: dict,
                                  prefix: str = ''):
        opt = self.opt.training

        pool_256_gt = self.pool_256(input_image)
        if self.opt.training.enable_G1:
            pred_imgs = self.pool_256(render_out['gen_imgs'])
            gt_imgs = pool_256_gt
        else:
            pred_imgs = render_out['gen_thumb_imgs']
            gt_imgs = self.pool_64(input_image)

        loss_2d_rec, loss_2d_rec_dict = self.loss_module.calc_2d_rec_loss(  # type: ignore
            pred_imgs,
            gt_imgs,
            pool_256_gt,  # got identity loss, 256 resolution
            opt,
            loss_dict=True)

        if self.opt.training.enable_G1 and opt.supervise_both_gen_imgs:
            loss_2d_rec_thumb, loss_2d_rec_dict_thumb = self.loss_module.calc_2d_rec_loss(  # type: ignore
                render_out['gen_thumb_imgs'],
                self.pool_64(input_image),
                gt_pool(input_image),  # got identity loss, 256 resolution
                opt,
                loss_dict=True)
            loss_2d_rec = loss_2d_rec + loss_2d_rec_thumb
            loss_2d_rec_dict.update({'thumb_rec_loss': loss_2d_rec_thumb})

        return loss_2d_rec, loss_2d_rec_dict

    def _calculate_shape_rec_loss(self,
                                  random_3d_sample_batch: dict,
                                  render_out: dict,
                                  prefix: str = ''):
        opt = self.opt.training

        pred_shape = {
            'uniform_points_sdf':
            render_out['uniform_pts_rec'],  # ?????. VAR NAMING !!! 
            'surface_sdf':
            render_out['xyz_rec'],
            'surface_eikonal_term':
            render_out['xyz_rec_eikonal_term']
            if opt.return_surface_eikonal else None,
            'eikonal_term':
            render_out['eikonal_term'],  # * to update
        }
        gt_shape = {
            'uniform_points_sdf':
            random_3d_sample_batch['uniform_points_sdf'],
            'surface_eikonal_term':
            random_3d_sample_batch['surface_eikonal_term']
            if opt.return_surface_eikonal else None,
        }

        # todo, to merge
        if opt.fg_mask:  # * todo
            for shape in (pred_shape, gt_shape):  # * mask invalid uniform pts
                shape['uniform_points_sdf'] = shape[
                    'uniform_points_sdf'] * random_3d_sample_batch[
                        'uniform_points_valid_mask']
                if opt.return_surface_eikonal:
                    shape['surface_eikonal_term'] = shape[
                        'surface_eikonal_term'] * random_3d_sample_batch['mask']

            pred_shape['surface_sdf'] = pred_shape[
                'surface_sdf'] * random_3d_sample_batch[
                    'mask']  # * mask invalid surf pts

        loss_shape, loss_shape_dict = self.loss_module.calc_shape_rec_loss(  # type: ignore
            pred_shape,
            gt_shape,
            self.device,
            supervise_sdf=True,
            supervise_surface=True,
            supervise_surface_normal=True,
            supervise_eikonal=True)  # todo, update eional equation

        return loss_shape, loss_shape_dict

    def _train_discriminator_step(self):
        opt = self.opt.training
        if self.opt.training.fixedD:
            return dict(d=torch.tensor(0.).to(self.device))

        self._toggle_adversarial_loss('d_step')
        requires_grad(self.generator, False)  # but pts need grad
        self.zero_grad()

        # prep fake imgs

        synthetic_data_sample = self.synthetic_data_sample()
        curr_fake_imgs, geometry_sample, rand_cam_settings = (
            synthetic_data_sample[k]
            for k in ('fake_imgs', 'sample_batch', 'cam_settings'))

        render_out = self.image2image(curr_fake_imgs, rand_cam_settings)

        # sample synthetic images
        loss_dict = {}
        multires_imgs: dict = next(self.train_loader)
        # todo, add Decoder support later

        if self.opt.training.enable_G1:
            real_imgs = multires_imgs['img'].to(self.device)
            # real_imgs = self.pool_256(multires_imgs['img'].to(self.device) )
            # gen_imgs = render_out['gen_imgs']

            if self.opt.training.D_aligned_res:
                aligned_res = render_out['que_render_out']['aligned_res']
                gt_res = render_out['que_render_out']['res_gt']
                gen_imgs = torch.cat(
                    (self.pool_256(render_out['gen_imgs']), aligned_res),
                    dim=1)
                real_imgs = torch.cat((real_imgs, gt_res), dim=1)
            else:
                # gen_imgs = self.pool_256(render_out['gen_imgs'])
                gen_imgs = render_out['gen_imgs']

            d_regularize = self._iter % opt.d_reg_every == 0
            if d_regularize:
                real_imgs.requires_grad = True

            for j in range(0, opt.batch, opt.chunk):
                curr_real_imgs = real_imgs[j:j + opt.chunk]
                curr_gen_imgs = gen_imgs[j:j + opt.chunk]

                # print(curr_gen_imgs.shape,curr_real_imgs.shape, flush=True)
                fake_pred = self.discriminator(curr_gen_imgs.detach())
                real_pred = self.discriminator(curr_real_imgs)

                # print(fake_pred.requires_grad, real_pred.requires_grad)

                d_gan_loss = d_logistic_loss(real_pred, fake_pred)

                if d_regularize:
                    grad_penalty = d_r1_loss(real_pred, curr_real_imgs)
                    r1_loss = opt.r1 * 0.5 * grad_penalty * opt.d_reg_every
                else:
                    r1_loss = torch.tensor(0.0, device=self.device)

                # d_loss = (d_gan_loss + r1_loss) * self.opt.training.discriminator_lambda
                d_loss = d_gan_loss * self.opt.training.discriminator_lambda + r1_loss
                d_loss.backward()

        else:
            real_imgs = multires_imgs['thumb_img'].to(self.device)
            gen_imgs = render_out['gen_thumb_imgs']
            gt_viewpoint = rand_cam_settings['viewpoint']
            fake_pred, fake_viewpoint_pred = self.discriminator(
                gen_imgs.detach())
            if opt.view_lambda:
                d_view_loss = opt.view_lambda * viewpoints_loss(  # todo
                    fake_viewpoint_pred, gt_viewpoint)
            real_imgs.requires_grad = True
            real_pred, _ = self.discriminator(real_imgs)
            loss_dict["d_view"] = d_view_loss

            d_gan_loss = d_logistic_loss(real_pred, fake_pred)  # todo
            grad_penalty = d_r1_loss(real_pred, real_imgs)  # todo
            r1_loss = opt.r1 * 0.5 * grad_penalty
            d_loss = d_gan_loss + r1_loss + d_view_loss
            # d_loss = (d_gan_loss + r1_loss + d_view_loss) * self.opt.training.discriminator_lambda
            d_loss = d_gan_loss * self.opt.training.discriminator_lambda + r1_loss + d_view_loss
            d_loss.backward()

        self.optimizer_d.step()

        loss_dict["d"] = d_gan_loss
        loss_dict["r1"] = r1_loss
        loss_dict["real_score"] = real_pred.mean()
        loss_dict["fake_score"] = fake_pred.mean()

        self._toggle_adversarial_loss('e_step')  # toggle back

        return loss_dict

    def _toggle_adversarial_loss(self, step='d_step'):

        if step == 'd_step':
            for k, v in self.network.items():
                requires_grad(v, False)
            requires_grad(self.g_module.renderer.network.netLocal, False)

            requires_grad(self.discriminator, True)

        elif step == 'e_step':
            for k, v in self.network.items():
                if k != 'encoder':  # D loss only in the final stage, where E0 is fixed
                    requires_grad(v, True)
            requires_grad(self.g_module.renderer.network.netLocal, True)

            requires_grad(self.discriminator, False)

    # todo, rewrite real-part
    def _compute_loss(self,
                      gt_imgs,
                      render_out,
                      geometry_sample=None,
                      prefix='',
                      lms=None,
                      aligned_res=None):
        # prep
        opt = self.opt.training
        loss_dict = {}
        loss = torch.tensor(0.0, device=self.device)

        # 2D reconstruction losses:
        loss_2d_rec, loss_2d_rec_dict = self._calculate_pixel_rec_loss(
            gt_imgs, render_out, prefix)
        loss += loss_2d_rec

        # gan loss
        if self.enable_adv_training:
            assert self.opt.training.adv_lambda > 0

            if self.opt.training.enable_G1:
                gen_imgs = render_out['gen_imgs']

                fake_pred = self.discriminator(gen_imgs)
            else:
                fake_pred, fake_viewpoint_pred = self.discriminator(
                    render_out['gen_thumb_imgs'])

            encoder_g_loss = g_nonsaturating_loss(fake_pred)
            if opt.view_lambda > 0 and not self.train_real:
                rand_cam_settings = render_out['pred_cam_settings']
                e_view_loss = opt.view_lambda * viewpoints_loss(
                    fake_viewpoint_pred, rand_cam_settings['viewpoint'])
            else:
                e_view_loss = torch.tensor(0.0, device=self.device)
                loss_dict["e_view"] = e_view_loss

            # add adaptive G weights, tricks from VQGAN | unleashing transformer
            if self.opt.training.adaptive_D_loss:
                assert self.opt.training.adv_lambda == 1
                last_layer = self.g_module.renderer.network.netLocal.image_filter.conv_last3.weight

                d_weight = calculate_adaptive_weight(loss_2d_rec,
                                                     encoder_g_loss,
                                                     last_layer, 1)

                # d_weight *= adopt_weight(1, self._iter, self.disc_start_step) # https://github.dev/samb-t/unleashing-transformers
                loss_dict["d_weight"] = d_weight
            else:
                d_weight = 1

            loss_dict["e"] = encoder_g_loss * d_weight
            loss_adversarial = loss_dict['e'] + loss_dict["e_view"]
            loss = loss + loss_adversarial * self.opt.training.adv_lambda

        # 3d losses
        if not opt.pix_sup_only:
            assert geometry_sample is not None, 'send geometry output here'
            loss_shape, loss_shape_dict = self._calculate_shape_rec_loss(
                geometry_sample, render_out, prefix)
            loss = loss + loss_shape
        else:
            loss_shape = 0
            loss_shape_dict = {}

        loss_dict.update({**loss_shape_dict, **loss_2d_rec_dict})
        return dict(loss=loss, loss_dict=loss_dict)

    @torch.no_grad()
    def render_latent_multiview(
            self,
            latent_in: list,
            input_is_latent: bool,
            # iter_suffix: int,
            trajectory: torch.Tensor = None,
            # img2render='gen_thumb_imgs',
            render_video=False):

        opt = self.opt.inference
        batch = latent_in[0].shape[0]
        assert batch == 1, 'batch multi-view not supported yet'

        if trajectory is None:
            trajectory = self.locations

        all_rgb = []
        all_depth_mesh_img = []
        num_frames = trajectory.shape[0]

        # * inference multi-view recosntructions
        for j in range(0, num_frames):
            cam_settings = self._cam_locations_2_cam_settings(
                batch, trajectory[j:j + 1])
            render_out = self.latent2image(latent_in,
                                           cam_settings=cam_settings,
                                           input_is_latent=input_is_latent)
            rgb = render_out[img2render]
            # Convert RGB from [-1, 1] to [0,255]
            rgb = 127.5 * (rgb.clamp(-1, 1).permute(0, 2, 3, 1).cpu().numpy() +
                           1)
            all_rgb.append(rgb[0])

            if not opt.no_surface_renderings:
                # B H W 3, H=512
                mesh_image = self.render_depth_mesh(latent_in,
                                                    True,
                                                    trajectory[j],
                                                    cam_settings=cam_settings)
                all_depth_mesh_img.append(mesh_image[0])  # remove batch dim

        # todo, add normal map
        if not render_video:
            # save gifo
            all_rgb = np.concatenate(all_rgb, axis=-2)  # in w dimension, bhw3
            all_depth_mesh_img = np.concatenate(all_depth_mesh_img, axis=-2)

            all_rgb = mmcv.rgb2bgr(all_rgb)  # for mmwrite tradition
            all_depth_mesh_img = mmcv.rgb2bgr(
                all_depth_mesh_img)  # for mmwrite tradition

            mmcv.imwrite(
                all_rgb,
                os.path.join(self.results_dst_dir, self.mode, 'images',
                             f'{iter_suffix}_rgb.png'))
            mmcv.imwrite(
                all_depth_mesh_img,
                os.path.join(self.results_dst_dir, self.mode, 'images',
                             f'{iter_suffix}_depth.png'))

        else:
            suffix = '_azim' if opt.azim_video else '_elipsoid'
            video_filename = 'video/{}_{}.mp4'.format(iter_suffix, suffix)
            writer = skvideo.io.FFmpegWriter(
                os.path.join(self.results_dst_dir, self.mode, video_filename),
                outputdict={
                    '-pix_fmt': 'yuv420p',
                    '-crf': '18',
                    #  '-crf': '10'
                })
            if not opt.no_surface_renderings:
                depth_video_filename = 'video/depth_{}_{}.mp4'.format(
                    iter_suffix, suffix)
                depth_writer = skvideo.io.FFmpegWriter(os.path.join(
                    self.results_dst_dir, self.mode, depth_video_filename),
                                                       outputdict={
                                                           '-pix_fmt':
                                                           'yuv420p',
                                                       })

            for frame_idx in len(all_rgb):
                writer.writeFrame(all_rgb[frame_idx])
                if not opt.no_surface_renderings:
                    depth_writer.writeFrame(all_depth_mesh_img[frame_idx])

            writer.close()
            if not opt.no_surface_renderings:
                depth_writer.close()

    @torch.no_grad()
    def latent2surface(
            self,
            latent_in: list,
            input_is_latent: bool,
            cam_pose: torch.Tensor = None,  # todo, leave one.
            local_condition: torch.Tensor = None,
            cam_settings: dict = None,
            return_mesh: bool = False,
            pti_generator=None):
        torch.cuda.empty_cache()
        opt = self.opt.training
        batch = latent_in[0].shape[0]

        if cam_settings is None:
            cam_settings = self._cam_locations_2_cam_settings(batch, cam_pose)

        scale = self.surface_g_ema.renderer.out_im_res / self.g_module.renderer.out_im_res
        surface_sample_focals = cam_settings['focal'] * scale

        if pti_generator is not None:
            surface_g_ema = pti_generator
        else:
            surface_g_ema = self.surface_g_ema

        surface_out = surface_g_ema(
            latent_in,
            cam_settings['poses'],
            surface_sample_focals,
            cam_settings['near'],
            cam_settings['far'],
            truncation=opt.truncation_ratio if not input_is_latent else 1,
            truncation_latent=self.mean_latent,
            return_xyz=True,
            return_mesh=return_mesh,
            mesh_with_shading=return_mesh,
            input_is_latent=input_is_latent)
        torch.cuda.empty_cache()

        return surface_out

    @torch.no_grad()
    def marching_cube(self, sdf, cam_settings, surface_out=None):
        try:
            frostum_aligned_sdf = align_volume(sdf)
            marching_cubes_mesh = extract_mesh_with_marching_cubes(
                frostum_aligned_sdf[0:1])
        except ValueError:
            marching_cubes_mesh = None
            print('Marching cubes extraction failed.')
            print(
                'Please check whether the SDF values are all larger (or all smaller) than 0.'
            )

        if marching_cubes_mesh != None:

            marching_cube_root = os.path.join(self.results_dst_dir, self.mode,
                                              'marching_cubes_meshes')
            marching_cube_root_path = Path(marching_cube_root)
            marching_cube_root_path.mkdir(exist_ok=True, parents=True)

            curr_locations = cam_settings['viewpoint'].squeeze()
            loc_str = '_azim{}_elev{}'.format(
                int(curr_locations[0] * 180 / np.pi),
                int(curr_locations[1] * 180 / np.pi))

            marching_cubes_mesh_filename = os.path.join(
                marching_cube_root,
                'sample_{}_marching_cubes_mesh{}.obj'.format(
                    self._iter, loc_str))
            with open(marching_cubes_mesh_filename, 'w') as f:
                marching_cubes_mesh.export(f, file_type='obj')
            # print(marching_cubes_mesh_filename)
            return marching_cubes_mesh

    @torch.no_grad()
    def save_depth_mesh(self,
                        surface_out,
                        cam_settings,
                        mesh_saveroot=None,
                        mesh_saveprefix=None):
        """ render mesh using pytorch3d
        """
        if mesh_saveroot is None:
            mesh_saveroot = os.path.join(self.results_dst_dir, self.mode,
                                         'meshes')
        if mesh_saveprefix is None:
            mesh_saveprefix = '{}'.format(self._iter)

        mesh_save_path = os.path.join(mesh_saveroot, f'{mesh_saveprefix}.obj')
        if surface_out.get('mesh', None) is not None:
            with open(mesh_save_path, 'w') as f:
                surface_out['mesh'].export(f, file_type='obj')

        # B H W 3, H=512
        depth_image = self.render_depth_mesh(None,
                                             True,
                                             surface_out=surface_out,
                                             cam_settings=cam_settings)[0]

        depth_img_file_path = os.path.join(mesh_saveroot,
                                           f'{mesh_saveprefix}_depth.png')

        bgr_depth_image = mmcv.rgb2bgr(depth_image)
        mmcv.imwrite(bgr_depth_image, depth_img_file_path)

        return mesh_save_path

    def render_trimesh(self, trimesh_mesh, trajectory_location):
        """render trimesh give pose

        Args:
            mesh (trimesh): marching cube output
            trajectory_location (torch.Tensor): pose 

        Returns:
            torch.Tensor: rendered img
        """
        # convert trimesh to pytorch3d mesh
        py3d_mesh = Meshes(
            verts=[
                torch.from_numpy(np.asarray(trimesh_mesh.vertices)).to(
                    torch.float32).to(self.device)
            ],
            faces=[
                torch.from_numpy(np.asarray(trimesh_mesh.faces)).to(
                    torch.float32).to(self.device)
            ],
            textures=None,
            verts_normals=[
                torch.from_numpy(
                    np.copy(np.asarray(trimesh_mesh.vertex_normals))).to(
                        torch.float32).to(self.device)
            ],
        )

        vertex_colors = torch.from_numpy(trimesh_mesh.visual.vertex_colors).to(
            self.device)[..., :3].to(torch.float32)
        py3d_mesh = add_textures(py3d_mesh, vertex_colors=vertex_colors)

        cameras = create_cameras(
            azim=np.rad2deg(trajectory_location[0].cpu().numpy()),
            elev=np.rad2deg(trajectory_location[1].cpu().numpy()),
            fov=2 * self.opt.training.camera.fov,  # todo, hard coded
            dist=1,
            device=self.device)
        renderer = create_mesh_renderer(
            cameras,
            image_size=512,
            # light_location=((0.0, 1.0, 5.0), ),
            light_location=((0.0, 3.0, 5.0), ),  # for shapenet
            specular_color=((0.2, 0.2, 0.2), ),
            ambient_color=((0.1, 0.1, 0.1), ),
            diffuse_color=((0.65, .65, .65), ),
            device=self.device)

        mesh_image = 255 * renderer(py3d_mesh).cpu().numpy()
        mesh_image = mesh_image[..., :3]

        # Add depth frame to video
        return mesh_image

    def _load_ckpt(self, ckpt):
        # todo, move load ckpt here
        if ckpt is not None:
            print('load from ckpt, _load_ckpt()')
            if self.opt.training.adv_lambda > 0 and 'd' in ckpt:
                self.discriminator.load_state_dict(ckpt['d'])
                print('load D from ckpt')

            print('load from ckpt, _load_ckpt(), network')
            for k, v in self.network.items():
                if k in ckpt and k not in self.opt.training.ckpt_to_ignore:
                    strict_load = True
                    # if k == 'grid_align' and self.opt.training.aligner_norm_type != 'batch':
                    try:
                        v.load_state_dict(ckpt[k], strict_load)
                        print(f'loading {k} ckpt', flush=True)
                    except Exception as e:
                        # print(f'ignore {k} ckpt', flush=True)
                        print(f'failed to load and ignore {k} ckpt', e, 
                              flush=True)

                else:
                    print(f'ignore ckpt weight of {k}', flush=True)
            print('load from ckpt, _load_ckpt(), network done')
        else:
            print('ckpt is None')

    def _post_load_ckpt(self, ckpt):
        if ckpt is not None:
            del ckpt
            refresh_cache()

    def _grad_flags(self):
        requires_grad(self.generator, False)
        opt_training = self.opt.training
        opt = self.opt

        # D
        if self.enable_adv_training:
            requires_grad(self.discriminator, self.train_discriminator)  # L

        # E0
        if self.opt.training.encoder_type == 'HybridGradualStyleEncoder_V2':
            requires_grad(self.e_module.input_layer,
                          not opt_training.E_backbone_false)  # L
            requires_grad(self.e_module.body,
                          not opt_training.E_backbone_false)  # L
            requires_grad(self.e_module.latlayer64,
                          not opt_training.E_backbone_false)  # L
            requires_grad(self.e_module.latlayer128,
                          not opt_training.E_backbone_false)  # L
            requires_grad(self.e_module.latlayer256,
                          not opt_training.E_backbone_false)  # L
            requires_grad(self.e_module.styles_pigan,
                          not opt_training.E_g_grad_false)  # G
            # decoder
            if self.opt.training.full_pipeline and not opt_training.disable_decoder_fpn:
                if self.opt.training.enable_G1:
                    requires_grad(self.e_module.styles_stylegan,
                                  not opt_training.E_d_grad_false)  # D
                else:
                    requires_grad(self.e_module.styles_stylegan, False)  # D
        elif self.opt.training.encoder_type == 'BackboneEncoderRenderer':
            requires_grad(self.e_module)

        else:  # DiscriminatorEncoder.
            requires_grad(self.e_module.final_conv,
                          not opt_training.E_backbone_false)  # L
            requires_grad(self.e_module.convs,
                          not opt_training.E_backbone_false)  # L
            requires_grad(self.e_module.wplus_latents_pred_conv,
                          not opt_training.E_g_grad_false)  # E0

        # E1
        if self.opt.rendering.enable_local_model:
            requires_grad(self.g_module.renderer.network.netLocal,
                          not opt_training.E_l_grad_false)
            if not opt.training.fix_renderer:  # load pretrained netLocal and modulation layers
                local_params = []
                local_params += list(
                    self.g_module.renderer.network.netLocal.parameters())

                # use modulation to add local features
                if opt.rendering.L_pred_tex_modulations:
                    requires_grad(
                        self.g_module.renderer.network.netLocal.
                        local_feat_to_tex_modulations_linear,
                        opt.rendering.L_pred_tex_modulations)
                if opt.rendering.L_pred_geo_modulations:
                    requires_grad(
                        self.g_module.renderer.network.netLocal.
                        local_feat_to_geo_modulations_linear,
                        opt.rendering.L_pred_geo_modulations)

    def _get_trainable_parmas(self):
        # shall be called before self.dist_prepare()

        opt = self.opt
        param_groups = []

        self._grad_flags()

        # handle Local Encoder grads
        if opt.rendering.enable_local_model:
            if opt.training.fix_renderer:  # load pretrained netLocal and modulation layers
                requires_grad(self.g_module.renderer.network.netLocal, False)
            else:  # add netLocal & modulation layers to the params
                requires_grad(self.g_module.renderer.network.netLocal, True)
                local_params = []

                for name, param in self.g_module.renderer.network.netLocal.named_parameters():
                    if 'local_feat_to_tex_modulations_linear' not in name:
                        local_params.append(param)
                        # print(name)

                local_params = (param for param in local_params)

                local_pifu_optim_group = {
                    'name': 'pifu',
                    'params': local_params,
                    'lr': self.opt.training.lr
                }
                param_groups.append(local_pifu_optim_group)

                local_pifu_linear_optim_group = {
                    'name': 'pifu_local_feat_to_tex_modulations_linear',
                    'params': self.g_module.renderer.network.netLocal.local_feat_to_tex_modulations_linear.parameters(),
                    'lr': self.opt.training.lr*0.2 # 1e-5
                }
                param_groups.append(local_pifu_linear_optim_group)

            # todo, add to e-params

        encoder_trainable_param = get_trainable_parameter(self.e_module)
        if len(encoder_trainable_param) > 0:
            encoder_trainable_param = (param
                                       for param in encoder_trainable_param)
            encoder_optim_group = {
                'name': 'encoder',
                'params': encoder_trainable_param,
                'lr': self.opt.training.lr
            }
            param_groups.append(encoder_optim_group)

        return param_groups

    def _build_model(self):  # build other models
        # create E0
        self.network.update(dict(encoder=self.encoder))

        self.lms_predictor = None
        self.heatmap_loss = None

        # create D
        self.train_discriminator = self.opt.training.discriminator_lambda > 0
        self.enable_adv_training = self.opt.training.adv_lambda > 0

        if self.train_discriminator:
            if not self.opt.training.enable_G1:
                self.discriminator = self.volume_discriminator
            else:
                assert self.discriminator is not None
            self.network.update(dict(discriminator=self.discriminator))  #

    def setup_optimizers(self):
        train_opt = self.opt.training
        param_groups = self._get_trainable_parmas()

        if train_opt.optim_name == 'adam':
            optimizer_e = torch.optim.Adam(param_groups)
        else:
            optimizer_e = Ranger(param_groups)

        self.optimizer_e = optimizer_e
        # self.optimizers.append(self.optimizer_e)
        self.optimizers.update(dict(optimizer_e=self.optimizer_e))

        if self.train_discriminator:  # follow tradition.
            d_reg_ratio = self.opt.training.d_reg_every / (
                self.opt.training.d_reg_every + 1)
            d_optim_group = {
                'name': 'discriminator',
                'params': self.discriminator.parameters(),
                'lr': self.opt.training.lr * d_reg_ratio,
                'betas': (0**d_reg_ratio, 0.99**d_reg_ratio),
            }
            self.optimizer_d = torch.optim.Adam([d_optim_group])
            # self.optimizers.append(self.optimizer_d)
            self.optimizers.update(dict(optimizer_d=self.optimizer_d))

        if get_rank() == 0:
            print('optimization param_groups: {}'.format(
                [param_group['name'] for param_group in param_groups]))

    def _dist_prepare(self):
        # set distributed models
        opt = self.opt.training
        if opt.distributed:
            if self.opt.rendering.enable_local_model:
                self.generator.renderer.network.netLocal = nn.SyncBatchNorm.convert_sync_batchnorm(self.generator.renderer.network.netLocal)
                print("using syncbn for pifu network")
            self.generator = nn.parallel.DistributedDataParallel(
                self.generator,
                device_ids=[opt.local_rank],
                output_device=opt.local_rank,
                find_unused_parameters=
                # False,  # decoder may not be engaged in the training.
                True,  # decoder may not be engaged in the training.
                broadcast_buffers=False)
            self.loss_class = nn.parallel.DistributedDataParallel(
                self.loss_class,
                device_ids=[opt.local_rank],
                output_device=opt.local_rank,
                broadcast_buffers=False,
                find_unused_parameters=False)
            self.volume_discriminator = nn.parallel.DistributedDataParallel(
                self.volume_discriminator,
                device_ids=[opt.local_rank],
                output_device=opt.local_rank,
                broadcast_buffers=False,
                find_unused_parameters=False)
            if self.lms_predictor is not None:
                self.lms_predictor = torch.nn.parallel.DistributedDataParallel(
                    self.lms_predictor,
                    device_ids=[opt.local_rank],
                    output_device=opt.local_rank,
                    broadcast_buffers=False,
                    find_unused_parameters=False)

            self.encoder = nn.parallel.DistributedDataParallel(
                self.encoder,
                device_ids=[opt.local_rank],
                output_device=opt.local_rank,
                broadcast_buffers=False,
                find_unused_parameters=True)

            if self.enable_adv_training:
                self.discriminator = nn.parallel.DistributedDataParallel(
                    self.discriminator,
                    device_ids=[opt.local_rank],
                    output_device=opt.local_rank,
                    broadcast_buffers=False,
                    find_unused_parameters=False)
            else:
                self.discriminator = self.volume_discriminator

        # * for saving
        if opt.distributed:
            self.g_module = self.generator.module
            self.d_module = self.discriminator.module
            self.e_module = self.encoder.module
            self.loss_module = self.loss_class.module

            if self.lms_predictor is not None:
                self.lms_predictor_module = self.lms_predictor.module
            else:
                self.lms_predictor_module = None
        else:
            self.lms_predictor_module = self.lms_predictor
            self.g_module = self.generator
            self.d_module = self.discriminator
            self.e_module = self.encoder
            self.loss_module = self.loss_class

        # self.models = [self.discriminator, self.encoder]
        self.generator.eval()  # todo, move L out

    def train_mode(self):
        self.mode = 'train'
        self.g_module.renderer.test = False
        # for model in self.models:
        #     if model is not None:
        #         model.train()

        for _, v in self.network.items():
            v.train()

    def eval_mode(self):
        self.mode = 'val'
        self.g_module.renderer.test = True

        for _, v in self.network.items():
            v.eval()

    @staticmethod
    @torch.no_grad()
    def normalize(tensor: torch.Tensor, value_range=None):
        if value_range is not None:
            assert isinstance(
                value_range, tuple
            ), "value_range has to be a tuple (min, max) if specified. min and max are numbers"

        tensor = tensor.clone()

        def norm_ip(img, low, high):
            img.clamp_(min=low, max=high)
            img.sub_(low).div_(max(high - low, 1e-5))

        def norm_range(t, value_range=None):
            if value_range is not None:
                norm_ip(t, value_range[0], value_range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        norm_range(tensor, value_range)
        return tensor

    @torch.no_grad()
    def evaluateTrajectory(self):
        self.eval_mode()

        # * load trajectory dataset
        opt = self.opt.inference
        trajectory_root = Path(opt.trajectory_gt_root)
        trajectory_dataset = [
            str(path) for path in sorted(trajectory_root.rglob('seed1234_id*'))
        ]  # first 100, for test.
        # trajectory_loader = iter(trajectory_dataset)

        chunk = 1  # shared
        trajectory = self.create_trajectory(250)

        # output metrics (and video if needed)
        val_imgsave_root_dir = Path(
            os.path.join(self.opt.results_dst_dir, self.mode, str(self._iter),
                         'trajectory'))
        print(
            'trajectory validation save dir: {}'.format(val_imgsave_root_dir))
        val_imgsave_root_dir.mkdir(parents=True, exist_ok=True)

        # * reconstruct the first view as reference view
        for idx, trajectory_dir in enumerate(tqdm(trajectory_dataset)):
            # if idx > 0:
            #     break

            id_name = Path(trajectory_dir).name

            suffix = '_azim' if opt.azim_video else '_elipsoid'
            video_dst_dir = os.path.join(self.results_dst_dir, self.mode,
                                         'trajectory_videos', str(id_name))
            Path(video_dst_dir).mkdir(parents=True, exist_ok=True)

            # create writers
            video_filename = 'sample_video_{}.mp4'.format(suffix)
            writer = skvideo.io.FFmpegWriter(os.path.join(
                video_dst_dir, video_filename),
                                             outputdict={
                                                 '-pix_fmt': 'yuv420p',
                                             })

            # load trajectory gt
            instance_trajectory_dataset = ImagesDatasetEval(
                trajectory_dir, img_name_order=True)
            instance_trajectory_loader = DataLoader(
                instance_trajectory_dataset, batch_size=251)
            trajectory_image = next(iter(instance_trajectory_loader))  # 251, 3

            trajectory_ref_view = self.pool_256(
                trajectory_image['image'][0:1]).to(self.device)
            trajectory_que_view = self.pool_256(
                trajectory_image['image'][1:251]).to(self.device)

            # ! add optimization-based method evaluation

            if self.opt.projection.inference_projection_validation:
                # w_inv_root = Path(self.opt.projection.w_inversion_root)
                inversed_w_latent_in = Path(
                    self.opt.projection.w_inversion_root
                ) / id_name / 'latent_in.pt'
                assert inversed_w_latent_in.exists()
                inversed_latent = torch.load(
                    inversed_w_latent_in,
                    map_location=lambda storage, loc: storage)
                latent_in = inversed_latent['latent_in']
                pred_latents = [latent.to(self.device) for latent in latent_in]

                if self.opt.projection.PTI:
                    pti_g_state = inversed_latent['g']
                    self.g_module.load_state_dict(pti_g_state, True)
                    # self.surface_g_ema.load_state_dict(pti_g_state, False)
                    print('load PTI weights')

                thrumb_trajectory_ref_view = self.pool_64(
                    trajectory_image['image'][0:1]).to(self.device)

                pred_cam_settings = self.image2camsettings(
                    thrumb_trajectory_ref_view)

                ref_imgs_info = self.latent2image(pred_latents,
                                                  pred_cam_settings)

            else:

                # ref view info
                ref_imgs_info = self.encode_ref_images(trajectory_ref_view)
                pred_latents = ref_imgs_info['pred_latents']

            if self.opt.inference.editing_inference:
                editing_boundary_scale_list = self.opt.inference.editing_boundary_scale
                edit_code_ret = self.edit_code(pred_latents,
                                               editing_boundary_scale_list)
                que_latents = edit_code_ret['edited_pred_latents']
            else:
                que_latents = pred_latents

            # loss
            trajectories_loss_list = []
            trajectory_loss_list = []

            # * for loop render RGB and Depth. Start from Depth first.
            for j in range(0, trajectory.shape[0], chunk):
                torch.cuda.empty_cache()
                # gt_imgs = trajectory_gt['image'][j:j+chunk].to(self.device)
                gt_imgs = trajectory_que_view[j:j + chunk]

                chunk_trajectory = trajectory[
                    j:j + chunk]  # currently only 1 supported
                chunk_cam_settings = self._cam_locations_2_cam_settings(
                    1, chunk_trajectory)

                que_info = self.latent2image(
                    # pred_latents,
                    que_latents,
                    chunk_cam_settings,
                    geometry_sample=None,  # just study pixel sup
                    local_condition=None,
                    sample_with_renderer=True,
                    input_is_latent=True)  # add later

                # * cross recosntruction
                que_render_ref_out = self.que_render_given_ref(
                    ref_imgs_info, que_info)  # cross reconstruction

                res_render_out = que_render_ref_out['res_render_out']
                pred_imgs = self.pool_256(res_render_out['gen_imgs'])

                writer.writeFrame(Tensor2Array(pred_imgs))

                _, loss_2d_rec_dict = self.loss_module.calc_2d_rec_loss(  # type: ignore
                    pred_imgs,
                    gt_imgs,
                    gt_imgs,
                    self.opt.training,
                    loss_dict=True,
                    mode='val')  # 21.9G
                trajectory_loss_list.append(loss_2d_rec_dict)

            trajectories_loss_list.extend(
                trajectory_loss_list)  # just do mean calculation here.
        # todo, merge dict output overall performance
        # todo, output json to local.

        all_scores = {}  # todo, defaultdict
        mean_all_scores = {}
        for loss_dict in trajectories_loss_list:  # type: ignore
            for k, v in loss_dict.items():
                v = v.item()
                if k not in all_scores:
                    # all_scores[f'{k}_val'] = [v]
                    all_scores[k] = [v]
                else:
                    all_scores[k].append(v)

        for k, v in all_scores.items():
            mean = np.mean(v)
            std = np.std(v)
            if k in ['loss_lpis', 'loss_ssim']:
                mean = 1 - mean
            result_str = '{} average loss is {:.4f} +- {:.4f}'.format(
                k, mean, std)
            mean_all_scores[k] = mean
            print(result_str)

        # todo, add dataset header
        with open(os.path.join(str(val_imgsave_root_dir), 'scores.json'),
                  'a') as f:
            json.dump({**all_scores, 'iter': self._iter}, f)

    @torch.no_grad()
    def post_process_depthMesh(self, cv2_img, render_out, surface_out=None):
        assert cv2_img.ndim == 3
        assert isinstance(cv2_img, np.ndarray)
        # 2d lms
        # pred_cam_settings = render_out['pred_cam_settings']
        # * transfer back to frontal view for lms query, avoid lms miss & follow wu et al. 2020
        canonical_render_out = self.latent2frontal_image(
            render_out['pred_latents'], True)
        rec_thumb_cv2_img = vis_tensor(self.pool_256(
            canonical_render_out['gen_thumb_imgs']),
                                       normalize=True)
        rec_thumb_cv2_img = plt_2_cv2(rec_thumb_cv2_img)
        landmarks = self.lms_predictor.get_landmarks(  # type: ignore
            rec_thumb_cv2_img)  # scale 256.
        # rec_thumb_cv2_img = vis_tensor(self.pool_256(render_out['gen_thumb_imgs']), normalize=True)

        landmarks_7 = landmark_98_to_7(landmarks)

        # todo
        landmarks_7_imgs = visualize_alignment(
            cv2_img, [landmarks_7])  # [0,1] for grid visualization

        # 2d lms -> 128 resolution
        landmarks_7_128 = landmarks_7 / 2
        # print(landmarks_7_128)
        landmarks_7_128_Tensor = torch.from_numpy(landmarks_7_128).long().cpu()

        frontal_cam_settings = self._cam_locations_2_cam_settings(
            batch=1, cam_locations=self.frontal_location)

        # marching cubes. always use the canonical pose.
        if surface_out is None:
            surface_out = self.latent2surface(
                render_out['pred_latents'],
                True,
                # cam_settings=pred_cam_settings,
                cam_settings=frontal_cam_settings,  # type: ignore
                return_mesh=True,
                local_condition=None)

        # depth mesh
        depth_image, depth_mesh = self.render_depth_mesh(
            None,
            True,
            surface_out=surface_out,
            cam_settings=frontal_cam_settings,  # type: ignore
            return_mesh=True)
        # mesh lms

        kpts_3d = []
        for i in range(7):
            kpts_3d.append(surface_out['xyz'][0, :, landmarks_7_128_Tensor[
                i, 1], landmarks_7_128_Tensor[i, 0]].cpu())  # ! uv tradition
        kpts_3d = torch.stack(kpts_3d, 0).numpy()

        bgr_depth_image = mmcv.rgb2bgr(depth_image)
        # mmcv.imwrite(bgr_depth_image, depth_img_file_path)

        # face_mesh, kpts_3d = normalize_mesh(face_mesh, kpts_3d)

        # unproject and get 3d lms
        ret_dict = dict(
            landmarks_7_imgs=landmarks_7_imgs / 255,  # type: ignore
            bgr_depth_image=bgr_depth_image,
            # landmarks_7_imgs_rec=landmarks_7_imgs_rec/255,
            landmarks_7_128_Tensor=landmarks_7_128_Tensor,
            kpts_3d=kpts_3d,
            face_mesh=depth_mesh,
        )

        return ret_dict

    @torch.no_grad()
    def align_img(self, cv2_img: torch.Tensor):

        # align with ffhq landmarks, return the aligned img (in np array)
        assert cv2_img.shape[0] == 1
        cv2_img = cv2_img[0].cpu().numpy()
        aligned_cv2_img = crop_one_img(
            self.FaceRestoreHelper,  # type: ignore
            cv2_img)[0]  # H W 3
        aligned_img_rgb = aligned_cv2_img[..., ::-1] / 255

        # transform to the tensor, [-1,1]
        aligned_rgb_Tensor = self.transform(aligned_img_rgb).to(
            self.device).unsqueeze(0)
        return aligned_cv2_img, aligned_rgb_Tensor

    @torch.no_grad()
    def evaluate3D(self, mode='val'):
        self.eval_mode()
        ''' NOW validation / test
        '''
        # *. cg cfgs
        output_dir = {
            'val': os.path.join(self.results_dst_dir, 'NOW_validation'),
            'test': os.path.join(self.results_dst_dir, 'NOW_test'),
        }[mode]

        print(f'evaluation mode: {mode}',
              f'output_dir: {output_dir}',
              flush=True)

        os.makedirs(output_dir, exist_ok=True)
        savefolder = os.path.join(output_dir, f'step_{self._iter:08}')
        os.makedirs(savefolder, exist_ok=True)
        # self.deca.eval()

        # run now validation images
        dataset = NoWDataset(
            # scale=(self.opt.dataset.scale_min + self.opt.dataset.scale_max) /
            scale=(1.4 + 1.8) / 2,
            mode=mode,
            crop_size=256,
            normalize_img=True)
        dataloader = DataLoader(
            dataset,
            # batch_size=8,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last=False)

        # faces = self.deca.flame.faces_tensor.cpu().numpy()
        for i, batch in enumerate(tqdm(dataloader, desc='now evaluation')):
            torch.cuda.empty_cache()
            # batch size = 1 here
            imagename = batch['imagename']
            try:
                aligned_cv2_img, aligned_rgb_Tensor_images = self.align_img(
                    batch['cv2_image'])  # 1 3 H W
            except:
                print(imagename)
                continue

            aligned_rgb_Tensor_images = aligned_rgb_Tensor_images.to(
                self.device)

            with torch.no_grad():
                render_out = self.image2image(
                    aligned_rgb_Tensor_images, render_surface=False
                )  # return surface has bug here, need fix.
                visdict = dict(input_imgs=aligned_rgb_Tensor_images)

                # get mesh & 3D LMS
                # post_process_out = post_process(aligned_cv2_img, render_out, imagename)
                post_process_out = self.post_process_depthMesh(
                    aligned_cv2_img, render_out)
                visdict.update(
                    dict(gen_imgs=render_out['gen_imgs'],
                         lms_7_imgs=torch.from_numpy(
                             post_process_out['landmarks_7_imgs']).permute(
                                 2, 0, 1).unsqueeze(0),
                         gen_thumb_imgs=render_out['gen_thumb_imgs']))

            assert aligned_rgb_Tensor_images.shape[0] == 1
            for k in range(aligned_rgb_Tensor_images.shape[0]):
                os.makedirs(os.path.join(savefolder, imagename[k]),
                            exist_ok=True)
                mesh, landmark_7 = post_process_out[
                    'face_mesh'], post_process_out['kpts_3d']

                # save 7 landmarks for alignment
                np.save(os.path.join(savefolder, f'{imagename[k]}.npy'),
                        landmark_7)

                # save mesh
                verts, faces = mesh.vertices, mesh.faces
                # todo, support batch inference
                util.write_obj(os.path.join(savefolder, f'{imagename[k]}.obj'),
                               vertices=verts,
                               faces=faces)
                # vis imgs
                for vis_name in visdict.keys():  #['inputs', 'rec_imgs']:
                    if isinstance(visdict[vis_name], torch.Tensor):
                        image = util.tensor2image(
                            self.pool_256(visdict[vis_name].squeeze()))
                    else:
                        image = visdict[vis_name]

                    name = imagename[k].split('/')[-1]
                    cv2.imwrite(
                        os.path.join(savefolder, imagename[k],
                                     name + '_' + vis_name + '.png'), image)

            # visualize results to check
            util.visualize_grid(visdict,
                                os.path.join(savefolder, f'{i}.png'),
                                size=256)

            ## then please run main.py in https://github.com/soubhiksanyal/now_evaluation, it will take around 30min to get the metric results
            # /mnt/lustre/yslan/Repo/Research/SIGA22/3dmetrics/now_evaluation
        self.train_mode()

    def _editing_code(self, wp_latent_codes: list, scale: dict):
        """w space editing.

        Args:
            wp_latent_codes (Tensor): predicted wp space latent code, in list.
            scale (dict): the scale to add the editing boundaries

        Returns:
            _type_: _description_
        """

        Bangs = scale['Bangs']
        Smiling = scale['Smailing']
        No_Beard = scale['No_Beard']
        num_samples = wp_latent_codes[0].shape[0]

        # new_codes_wp = wp_latent_codes.copy()
        new_codes_wp = wp_latent_codes[0].cpu().numpy().copy()

        boundaries = self.editing_boundaries
        ATTRS = self.editing_attrs
        space = self.opt.editing.space

        for i, attr_name in enumerate(ATTRS):
            boundary = boundaries[attr_name]
            if new_codes_wp.ndim == 3:  # wp space, should be True
                boundary = np.expand_dims(
                    boundary, 1)  # 1 1 256, for broadcasting add with B 9 256
            new_codes_wp += boundary * scale[attr_name]

        # * post processing to Tensor

        Tensor_new_codes_wp = torch.Tensor(new_codes_wp).to(self.device)

        # * todo, only edit the first few layers here
        Tensor_new_codes_wp_tuple = [Tensor_new_codes_wp, wp_latent_codes[1]]

        output_img_name = f'{space}_Bangs{Bangs}_Smile{Smiling}_Beard{No_Beard}.png'

        return Tensor_new_codes_wp_tuple, output_img_name

    @torch.inference_mode()
    def render_depth_mesh(self,
                          latent_in: list,
                          input_is_latent: bool,
                          trajectory_location: torch.Tensor = None,
                          cam_settings: dict = None,
                          surface_out: dict = None,
                          return_mesh=False,
                          filter_out_bg=True):
        """render depth geometry given a latent code and trajectory location. Surface out may already be available.

        Args:
            latent_in (list): _description_
            input_is_latent (bool): _description_
            trajectory_location (torch.Tensor, optional): _description_. Defaults to None.
            cam_settings (dict, optional): _description_. Defaults to None.
            surface_out (dict, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        torch.cuda.empty_cache()
        if surface_out is None:
            surface_out = self.latent2surface(latent_in,
                                              input_is_latent,
                                              trajectory_location,
                                              cam_settings=cam_settings)
        else:
            trajectory_location = cam_settings['viewpoint'].squeeze()

        # this is done to fit to RTX2080 RAM size (11GB)
        xyz = surface_out['xyz']
        if filter_out_bg:
            bg_mask = (surface_out['gen_thumb_imgs'] > 0.98).float()
            bg_mask = torch.nn.functional.interpolate(
                bg_mask, size=(512, 512)).cpu()  # B H W 3, for image mask
            bg_mask = blur(bg_mask).permute(0, 2, 3, 1).numpy()
        else:
            bg_mask = None

        del surface_out
        torch.cuda.empty_cache()

        # Render mesh for video
        depth_mesh = xyz2mesh(xyz)
        mesh = Meshes(
            verts=[
                torch.from_numpy(np.asarray(depth_mesh.vertices)).to(
                    torch.float32).to(self.device)
            ],
            faces=[
                torch.from_numpy(np.asarray(depth_mesh.faces)).to(
                    torch.float32).to(self.device)
            ],
            textures=None,
            verts_normals=[
                torch.from_numpy(np.copy(np.asarray(
                    depth_mesh.vertex_normals))).to(torch.float32).to(
                        self.device)
            ],
        )
        mesh = add_textures(mesh)
        cameras = create_cameras(
            azim=np.rad2deg(trajectory_location[0].cpu().numpy()),
            elev=np.rad2deg(trajectory_location[1].cpu().numpy()),
            fov=2 * self.opt.training.camera.fov,  # todo, hard coded
            dist=1,
            device=self.device)
        renderer = create_mesh_renderer(
            cameras,
            image_size=512,
            # light_location=((0.0, 1.0, 5.0), ),
            light_location=((0.0, 0.0, 5.0), ),
            specular_color=((0.2, 0.2, 0.2), ),
            ambient_color=((0.1, 0.1, 0.1), ),
            diffuse_color=((0.65, .65, .65), ),
            device=self.device)

        mesh_image = 255 * renderer(mesh).cpu().numpy()
        mesh_image_ = mesh_image[..., :3]

        if bg_mask is not None:
            ambient_mask = bg_mask
            # ambient_mask = bg_mask * np.ones_like(mesh_image) # 1 for background
            # st()
            mesh_image = mesh_image_ * (
                1 - ambient_mask) + ambient_mask * 255 * 0.5  # ambient color
            # mesh_image = mesh_image_ * (1-ambient_mask) + ambient_mask * 255 # ambient color
        else:
            mesh_image = mesh_image_

        if return_mesh:
            return mesh_image, depth_mesh
        else:
            # Add depth frame to video
            return mesh_image

    @torch.no_grad()
    def create_trajectory(self, num_frames=250):
        # Generate video trajectory
        trajectory = np.zeros((num_frames, 3), dtype=np.float32)
        opt = self.opt

        # set camera trajectory
        # sweep azimuth angles (4 seconds)
        if opt.inference.azim_video:
            t = np.linspace(0, 1, num_frames)
            elev = 0
            fov = opt.camera.fov
            if opt.camera.uniform:
                azim = opt.camera.azim * np.cos(t * 2 * np.pi)
            else:
                # azim = 1.5 * opt.camera.azim * np.cos(t * 2 * np.pi)
                azim = 1.5 * opt.camera.azim * np.cos(
                    t * 1 * np.pi)  # only one side

            trajectory[:num_frames, 0] = azim
            trajectory[:num_frames, 1] = elev
            trajectory[:num_frames, 2] = fov

        # elipsoid sweep (4 seconds)
        else:
            t = np.linspace(0, 1, num_frames)
            fov = opt.camera.fov  #+ 1 * np.sin(t * 2 * np.pi)
            if opt.camera.uniform:
                elev = opt.camera.elev / 2 + opt.camera.elev / 2 * np.sin(
                    t * 2 * np.pi)
                azim = opt.camera.azim * np.cos(t * 2 * np.pi)
            else:
                elev = 1.5 * opt.camera.elev * np.sin(t * 2 * np.pi)
                azim = 1.5 * opt.camera.azim * np.cos(t * 2 * np.pi)

            trajectory[:num_frames, 0] = azim
            trajectory[:num_frames, 1] = elev
            trajectory[:num_frames, 2] = fov

        trajectory = torch.from_numpy(trajectory).to(self.device)
        return trajectory

    def _load_editing_directions(self):
        boundary_root_path = Path(self.opt.inference.editing_boundary_dir)

        space = 'renderer'  # apply directions in the 3D rendering space.
        # ATTRS = ["Bangs", "Smiling", "No_Beard", "Young", "Eyeglasses"]
        ATTRS = ["Bangs", "Smiling", "No_Beard", "Young"]
        boundaries = {k: {} for k in ATTRS}

        for attr_name in ATTRS:
            for space in ('renderer', 'decoder'):
                # boundary_file_path = boundary_root_path / 'boundaries_cvpr23' / 'stylesdf' / f'{space}_{attr_name}/boundary.npy'
                boundary_file_path = boundary_root_path / f'{space}_{attr_name}/boundary.npy'
                boundary = np.load(boundary_file_path)
                boundaries[attr_name][space] = boundary
                del boundary

        self.boundaries = boundaries

        self.ATTRS = ATTRS
        print(ATTRS)

        print('init editing directions done.')
        self.load_editing_dir = True

    # def edit_code_2spaces(self,
    def edit_code(self,
                  pred_latents,
                  editing_boundary_scale_list: list = None):
        ATTRS = self.ATTRS

        if editing_boundary_scale_list is None:
            editing_boundary_scale_list = [0, 1, 0, 0, 0]

        editing_boundary_scale_dict = dict(  # add smile by default
            Bangs=editing_boundary_scale_list[0],
            Smiling=editing_boundary_scale_list[1],
            No_Beard=editing_boundary_scale_list[2],
            Young=editing_boundary_scale_list[3],
            Eyeglasses=editing_boundary_scale_list[4])

        edited_codes = []

        spaces = ('renderer', 'decoder')

        for idx in range(2):
            space = spaces[idx]
            pred_latent_code = pred_latents[idx]
            new_codes = pred_latent_code.cpu().numpy()

            for i, attr_name in enumerate(ATTRS):
                boundary = self.boundaries[attr_name][space]

                if new_codes.ndim == 3:
                    boundary = np.expand_dims(
                        boundary,
                        1)  # 1 1 256, for broadcasting add with B 9 256
                new_codes += boundary * editing_boundary_scale_dict[attr_name]

            edited_codes.append(new_codes)

        edited_pred_latents = [
            torch.Tensor(code).to(self.device) for code in edited_codes
        ]
        output_img_name = f"Bangs{editing_boundary_scale_dict['Bangs']}_Smile{editing_boundary_scale_dict['Smiling']}_Beard{editing_boundary_scale_dict['No_Beard']}.png"

        return dict(output_img_name=output_img_name,
                    edited_pred_latents=edited_pred_latents)

    def edit_code_renderer(
            self,
            # def edit_code(self,
            pred_latents,
            editing_boundary_scale_list: list = None):
        ATTRS = self.ATTRS
        print('using old edit code')

        if editing_boundary_scale_list is None:
            editing_boundary_scale_list = [0, 1, 0]

        editing_boundary_scale_dict = dict(  # add smile by default
            Bangs=editing_boundary_scale_list[0],
            Smiling=editing_boundary_scale_list[1],
            No_Beard=editing_boundary_scale_list[2],
            Young=editing_boundary_scale_list[3],
            Eyeglasses=editing_boundary_scale_list[4])

        w_latent_codes = pred_latents[0]
        new_codes_w = w_latent_codes.cpu().numpy().copy()

        # do editing
        for i, attr_name in enumerate(ATTRS):
            boundary = self.boundaries[attr_name]['renderer']
            if new_codes_w.ndim == 3:
                boundary = np.expand_dims(
                    boundary, 1)  # 1 1 256, for broadcasting add with B 9 256
            # new_codes_w += boundaries[attr_name] * eval(attr_name)
            # new_codes_w += self.boundaries[attr_name] * editing_boundary_scale_dict[attr_name]
            new_codes_w += boundary * editing_boundary_scale_dict[attr_name]

        # to tensor
        Tensor_new_codes = torch.Tensor(new_codes_w).to(self.device)
        edited_pred_latents = [Tensor_new_codes, pred_latents[1]]  # todo?

        output_img_name = f"Bangs{editing_boundary_scale_dict['Bangs']}_Smile{editing_boundary_scale_dict['Smiling']}_Beard{editing_boundary_scale_dict['No_Beard']}.png"

        return dict(output_img_name=output_img_name,
                    edited_pred_latents=edited_pred_latents)

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
            encoder_out = self.image2latents(input_imgs)

        if isinstance(encoder_out, dict):
            pred_latents = encoder_out['pred_latents']
        else:
            pred_latents = encoder_out

        # 1. get latents
        pred_latents = self.image2latents(input_imgs)

        # 2. get depth map
        if cam_settings is None:
            pred_cam_settings = self.image2camsettings(input_thumb_imgs)
        else:
            pred_cam_settings = cam_settings

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

        else:
            edit_render_out = None
            # depth = render_out['depth']  # B H W 1 1
            que_latents = pred_latents

        # !
        ref_imgs_info = dict(
            # **encoder_out,
            encoder_out=encoder_out,
            imgs=input_imgs,
            cam_settings=pred_cam_settings,
            #  global_render_out=render_out,
            edit_render_out=edit_render_out,
            pred_latents=que_latents)

        return ref_imgs_info

    def que_render_given_ref(
            self,
            ref_imgs_info: dict,
            que_info: dict,
            #  pred_cam_settings=None,
            **kwargs):
        # 1. query info needed
        # res_gt = ref_imgs_info['res_gt']
        if 'cam_settings' in que_info:
            que_cam_settings = que_info['cam_settings']
        else:
            que_cam_settings = ref_imgs_info['cam_settings']
        # ref_render_out = ref_imgs_info['global_render_out']

        # 2. get projected feats
        # que_wd_pts = que_info['points']
        # ref_calibs = ref_imgs_info['cam_settings']['calibs']

        # que_cam_settings = que_info['cam_settings']
        # que_calibs = que_info['cam_settings']['calibs']
        # pred_latents = ref_imgs_info['pred_latents']
        # que_cam_settings = ref_imgs_info['cam_settings']
        pred_latents = que_info['pred_latents']

        render_out = self.latent2image(pred_latents,
                                       que_cam_settings,
                                       geometry_sample=None)

        if not self.opt.training.enable_G1:
            render_out['gen_imgs'] = render_out['gen_thumb_imgs']

        que_img_for_vis = self.pool_256(render_out['gen_thumb_imgs'])
        if self.opt.training.full_pipeline:
            que_img_for_vis = torch.cat((
                que_img_for_vis,
                self.pool_256(render_out['gen_imgs']),
            ), -1)

        return dict(
            res_render_out=render_out,
            que_img_for_vis=que_img_for_vis,
        )

    def edit_images(self, images: torch.Tensor, ref_imgs_info):
        """directly use camera_settings in ref_imgs_info
        """
        images = images.to(self.device)
        if images.shape[-1] != 256:
            images = self.pool_256(images)

        # do reconstruction
        if not self.opt.projection.inference_projection_validation:
            ref_imgs_info = self.encode_ref_images(images)
        pred_latents = ref_imgs_info['pred_latents']


        editing_que_info = self.latent2image(
            pred_latents,
            ref_imgs_info['cam_settings'],
            geometry_sample=None,  # just study pixel sup
            local_condition=None,
            sample_with_renderer=True,
            input_is_latent=True)  # add later

        # do local feature align
        # todo, currently only the front layer code is edited. need to add editng on the decoder?
        que_render_ref_out = self.que_render_given_ref(
            ref_imgs_info, editing_que_info)  # cross reconstruction

        render_out = que_render_ref_out['res_render_out']

        # render_out.update(res_gt=res_gt)
        render_out['edit_gen_thumb_imgs'] = editing_que_info['gen_thumb_imgs']

        return render_out

    def render_video(self, images, id_name, *args, **kwargs):
        """images -> video of geometry and depth, 3D consistency check.

        Args:
            all_rgb (torch.Tensor): input images
            save_gif (bool, optional): save as gif? Defaults to False.
        """
        import skvideo.io
        import torch
        from tqdm import tqdm

        from project.utils.misc_utils import Tensor2Array

        torch.cuda.empty_cache()
        opt = self.opt.inference

        # get input
        images = self.pool_256(images).to(self.device)

        # encode ref view
        ref_imgs_info = self.encode_ref_images(images)
        # ref_render_out = ref_imgs_info['global_render_out']
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
        writer = skvideo.io.FFmpegWriter(os.path.join(video_dst_dir,
                                                      video_filename),
                                         outputdict={
                                             '-pix_fmt': 'yuv420p',
                                             '-crf': '18',
                                         })
        if not opt.no_surface_renderings:
            depth_video_filename = 'sample_depth_video_{}.mp4'.format(suffix)
            depth_writer = skvideo.io.FFmpegWriter(os.path.join(
                video_dst_dir, depth_video_filename),
                                                   outputdict={
                                                       '-pix_fmt': 'yuv420p',
                                                       '-crf': '1'
                                                   })
            if Path(os.path.join(video_dst_dir,
                                 depth_video_filename)).exists():
                print("ignore {}".format(
                    os.path.join(video_dst_dir, depth_video_filename)))
                return

        if self.opt.inference.editing_inference:
            editing_boundary_scale_list = self.opt.inference.editing_boundary_scale
            edit_code_ret = self.edit_code(pred_latents,
                                           editing_boundary_scale_list)
            que_latents = edit_code_ret['edited_pred_latents']
        else:
            que_latents = pred_latents

        # * for loop render RGB and Depth. Start from Depth first.
        if self.opt.inference.output_id_loss:
            id_loss_list = []
        for j in tqdm(range(0, trajectory.shape[0], chunk)):
            torch.cuda.empty_cache()
            if not j % self.opt.inference.video_interval == 0:
                continue  # for pick imgs

            chunk_trajectory = trajectory[j:j +
                                          chunk]  # currently only 1 supported
            chunk_cam_settings = self._cam_locations_2_cam_settings(
                1, chunk_trajectory)

            que_info = self.latent2image(
                que_latents,
                chunk_cam_settings,
                geometry_sample=None,  # just study pixel sup
                local_condition=None,
                sample_with_renderer=True,
                input_is_latent=True)  # add later

            # * cross recosntruction
            que_render_ref_out = self.que_render_given_ref(
                ref_imgs_info,
                que_info,
            )

            res_render_out = que_render_ref_out['res_render_out']

            output_frame = res_render_out['gen_imgs']

            if self.opt.inference.output_id_loss:
                id_loss_gt = images
                arcface_input = self.pool_256(res_render_out['gen_imgs'])
                loss_id, _, _ = self.criterionID(arcface_input, id_loss_gt,
                                                 id_loss_gt)
                id_loss_list.append(loss_id.item())

            if self.opt.inference.save_independent_img:
                utils.save_image(
                    output_frame,
                    Path(video_dst_dir) / f'{j}.png',
                    # Path(video_dst_dir) / f'{j}.jpg',
                    nrow=1,
                    normalize=True,
                    value_range=(-1, 1),
                )
            utils.save_image(
                que_render_ref_out['que_img_for_vis'],
                Path(video_dst_dir) / f'{j}images_for_vis.jpg',
                nrow=1,
                normalize=True,
                value_range=(-1, 1),
            )

            writer.writeFrame(Tensor2Array(output_frame))

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

        # todo, add dataset header
        if self.opt.inference.output_id_loss:
            with open(Path(video_dst_dir) / 'identity_scores.log', 'a') as f:
                f.write(f'id_loss_mean: {id_loss_list.mean()}')

        print(f'output dir: {os.path.join(video_dst_dir, video_filename)}')

    @torch.no_grad()
    def render_edit_video(
        self,
        images,
        id_name,
        # save_gif=False,
        suffix='',
        # res_gt=None,
        # num_frames=250,
        attr_idx=None,
        encoder_out=None,
    ):
        """render the video of the edited inversed codes.
        """

        # prepare editing
        torch.cuda.empty_cache()
        opt = self.opt.inference
        images = self.pool_256(images).to(self.device)
        chunk = 1

        trajectory = self.create_trajectory(self.opt.inference.video_frames)
        suffix = '_azim' if opt.azim_video else '_elipsoid'
        video_dst_dir = os.path.join(self.results_dst_dir, self.mode, 'videos',
                                     str(id_name))

        video_dst_dir_Path = Path(video_dst_dir)
        Path(video_dst_dir).mkdir(parents=True, exist_ok=True)

        # create writers
        video_filename = 'sample_video_{}.mp4'.format(suffix)
        if Path(video_filename).exists():
            return

        print(
            'output trajectory: ',
            os.path.join(video_dst_dir, video_filename),
        )

        writer = skvideo.io.FFmpegWriter(os.path.join(video_dst_dir,
                                                      video_filename),
                                         outputdict={
                                             '-pix_fmt': 'yuv420p',
                                             '-crf': '18'
                                         })
        if not opt.no_surface_renderings:
            depth_video_filename = 'sample_depth_video_{}.mp4'.format(suffix)
            depth_writer = skvideo.io.FFmpegWriter(os.path.join(
                video_dst_dir, depth_video_filename),
                                                   outputdict={
                                                       '-pix_fmt': 'yuv420p',
                                                       '-crf': '1'
                                                   })

        assert self.opt.inference.editing_inference

        # inverse the input images to get the pred_latents
        if not self.opt.inference.editing_inference:
            encoder_out = self.image2latents(images)
        else:
            assert encoder_out is not None

        if isinstance(encoder_out, dict):
            pred_latents = encoder_out['pred_latents']
        else:
            pred_latents = encoder_out

        if self.opt.editing.render_video_for_each_direction:
            assert attr_idx is not None  # * only select one attribute to edit
            editing_boundary_scale_array_end = np.zeros(
                len(self.opt.inference.editing_boundary_scale_upperbound))
            editing_boundary_scale_array_beg = np.zeros_like(
                editing_boundary_scale_array_end)

            if self.opt.inference.editing_relative_scale >= 0:  # relative editing scale between [0,1] scaled with the heauristic upper/lower bound
                editing_boundary_scale_array_beg[
                    attr_idx] = self.opt.inference.editing_boundary_scale_lowerbound[
                        attr_idx] * (
                            1 - self.opt.inference.editing_relative_scale
                        ) + self.opt.inference.editing_boundary_scale_upperbound[
                            attr_idx] * self.opt.inference.editing_relative_scale
                editing_boundary_scale_array_end[
                    attr_idx] = editing_boundary_scale_array_beg[attr_idx]
            else:  # linear interpolate the scales to render
                editing_boundary_scale_array_end[
                    attr_idx] = self.opt.inference.editing_boundary_scale_upperbound[
                        attr_idx]
                editing_boundary_scale_array_beg[
                    attr_idx] = self.opt.inference.editing_boundary_scale_lowerbound[
                        attr_idx]

        else:
            editing_boundary_scale_array_end = np.array(
                self.opt.inference.editing_boundary_scale_upperbound) # array of scalar

            editing_boundary_scale_array_beg = np.array(
                self.opt.inference.editing_boundary_scale_lowerbound)

        # * for loop render RGB and Depth. Start from Depth first.
        for j in tqdm(range(0, trajectory.shape[0], chunk)):
            torch.cuda.empty_cache()

            # get the interpolated editing boundary scale
            editing_boundary_scale_interp = (
                j / trajectory.shape[0]) * editing_boundary_scale_array_end + (
                    1 - (j / trajectory.shape[0])
                ) * editing_boundary_scale_array_beg
            editing_boundary_scale_list = editing_boundary_scale_interp.tolist(
            )
            print(editing_boundary_scale_list)

            # inverse and edit the input image
            ref_imgs_info = self.encode_ref_images(
                images,
                editing_boundary_scale_list=
                editing_boundary_scale_list,  # ! use defualt smiling to fix bug
            )
            pred_latents = ref_imgs_info['pred_latents']
            que_latents = pred_latents
            chunk_trajectory = trajectory[j:j +
                                          chunk]  # currently only 1 supported
            chunk_cam_settings = self._cam_locations_2_cam_settings(
                1, chunk_trajectory)

            # render with G0
            edit_render_out = self.latent2image(
                que_latents,
                chunk_cam_settings,
                geometry_sample=None,  # just study pixel sup
                local_condition=None,
                sample_with_renderer=True,
                input_is_latent=True)  # add later

            # render with G1
            que_render_ref_out = self.que_render_given_ref(
                ref_imgs_info, edit_render_out)  # cross reconstruction

            res_render_out = que_render_ref_out['res_render_out']
            output_frame = res_render_out['gen_imgs']

            # write video and close()
            writer.writeFrame(Tensor2Array(output_frame))

            if self.opt.inference.save_independent_img:
                # pred_img_save_path = metrics_root_dir / img_name

                utils.save_image(
                    output_frame,
                    video_dst_dir_Path / f'{j}.png',
                    nrow=1,
                    normalize=True,
                    value_range=(-1, 1),
                )

            # depth
            # todo, reuse rendered output here?
            if not opt.no_surface_renderings:
                mesh_image = self.render_depth_mesh(
                    que_latents,
                    True,
                    chunk_trajectory.squeeze(),
                    cam_settings=chunk_cam_settings)[0]
                depth_writer.writeFrame(mesh_image[:])

                bgr_depth_image = mmcv.rgb2bgr(mesh_image)
                mmcv.imwrite(
                    bgr_depth_image,
                    str(Path(video_dst_dir_Path) / f'{j}_depth.png', ))

        writer.close()
        if not opt.no_surface_renderings:
            depth_writer.close()

    # TODO, will not release the following functions
    @torch.no_grad()
    def evaluateTrajectoryPickID(self):
        # filter high performance trajectory with identity loss
        self.eval_mode()

        # 1. load all videos

        # 2. calculate id loss

        # 3. save metrics locally

        # * load trajectory dataset
        opt = self.opt.inference
        trajectory_root = Path(opt.trajectory_gt_root)
        trajectory_dataset = [
            str(path) for path in sorted(trajectory_root.rglob('seed1234_id*'))
        ]  # first 100, for test.
        # trajectory_loader = iter(trajectory_dataset)

        chunk = 1  # shared
        trajectory = self.create_trajectory(250)

        # output metrics (and video if needed)
        val_imgsave_root_dir = Path(
            os.path.join(self.opt.results_dst_dir, self.mode, str(self._iter),
                         'trajectory'))
        print(
            'trajectory validation save dir: {}'.format(val_imgsave_root_dir))
        val_imgsave_root_dir.mkdir(parents=True, exist_ok=True)

        # * reconstruct the first view as reference view
        for idx, trajectory_dir in enumerate(tqdm(trajectory_dataset)):
            # if idx > 1:
            #     break

            id_name = Path(trajectory_dir).name

            suffix = '_azim' if opt.azim_video else '_elipsoid'
            video_dst_dir = os.path.join(self.results_dst_dir, self.mode,
                                         'trajectory_videos', str(id_name))
            Path(video_dst_dir).mkdir(parents=True, exist_ok=True)

            # create writers
            video_filename = 'sample_video_{}.mp4'.format(suffix)
            writer = skvideo.io.FFmpegWriter(
                os.path.join(video_dst_dir, video_filename),
                outputdict={
                    '-pix_fmt': 'yuv420p',
                    #  '-crf': '10'
                })

            # load trajectory gt
            instance_trajectory_dataset = ImagesDatasetEval(
                trajectory_dir, img_name_order=True)
            instance_trajectory_loader = DataLoader(
                instance_trajectory_dataset, batch_size=251)
            trajectory_image = next(iter(instance_trajectory_loader))  # 250, 3

            trajectory_ref_view = self.pool_256(
                trajectory_image['image'][0:1]).to(self.device)
            trajectory_que_view = self.pool_256(
                trajectory_image['image'][1:251]).to(self.device)
            trajectory = trajectory[1:]

            # get reference input,

            # ref view info
            ref_imgs_info = self.encode_ref_images(trajectory_ref_view)
            pred_latents = ref_imgs_info['pred_latents']

            if self.opt.inference.editing_inference:
                editing_boundary_scale_list = self.opt.inference.editing_boundary_scale
                edit_code_ret = self.edit_code(pred_latents,
                                               editing_boundary_scale_list)
                que_latents = edit_code_ret['edited_pred_latents']
            else:
                que_latents = pred_latents

            # loss
            trajectories_loss_list = []
            trajectory_loss_list = []

            # * for loop render RGB and Depth. Start from Depth first.
            for j in range(0, trajectory.shape[0], chunk):
                torch.cuda.empty_cache()
                # gt_imgs = trajectory_gt['image'][j:j+chunk].to(self.device)
                gt_imgs = trajectory_que_view[j:j + chunk]

                chunk_trajectory = trajectory[
                    j:j + chunk]  # currently only 1 supported
                chunk_cam_settings = self._cam_locations_2_cam_settings(
                    1, chunk_trajectory)

                que_info = self.latent2image(
                    # pred_latents,
                    que_latents,
                    chunk_cam_settings,
                    geometry_sample=None,  # just study pixel sup
                    local_condition=None,
                    sample_with_renderer=True,
                    input_is_latent=True)  # add later

                # * cross recosntruction
                que_render_ref_out = self.que_render_given_ref(
                    ref_imgs_info, que_info)  # cross reconstruction

                res_render_out = que_render_ref_out['res_render_out']
                pred_imgs = self.pool_256(res_render_out['gen_imgs'])

                writer.writeFrame(Tensor2Array(pred_imgs))

                _, loss_2d_rec_dict = self.loss_module.calc_2d_rec_loss(  # type: ignore
                    pred_imgs,
                    gt_imgs,
                    gt_imgs,
                    self.opt.training,
                    loss_dict=True,
                    mode='val')  # 21.9G
                trajectory_loss_list.append(loss_2d_rec_dict)

            trajectories_loss_list.extend(
                trajectory_loss_list)  # just do mean calculation here.
        # todo, merge dict output overall performance
        # todo, output json to local.

        all_scores = {}  # todo, defaultdict
        mean_all_scores = {}
        for loss_dict in trajectories_loss_list:
            for k, v in loss_dict.items():
                v = v.item()
                if k not in all_scores:
                    # all_scores[f'{k}_val'] = [v]
                    all_scores[k] = [v]
                else:
                    all_scores[k].append(v)

        for k, v in all_scores.items():
            mean = np.mean(v)
            std = np.std(v)
            if k in ['loss_lpis', 'loss_ssim']:
                mean = 1 - mean
            result_str = '{} average loss is {:.4f} +- {:.4f}'.format(
                k, mean, std)
            mean_all_scores[k] = mean
            print(result_str)

        # todo, add dataset header
        with open(os.path.join(str(val_imgsave_root_dir), 'scores.json'),
                  'a') as f:
            json.dump({**all_scores, 'iter': self._iter}, f)

    # for outputing trajectory with video as the input
    @torch.inference_mode()
    def render_HDTF(self, eval_imgs=3000, mode='val'):

        # TODO, include identities from cli

        loader, val_imgsave_root_dir, metrics_root_dir = self.on_val_start(
            mode)

        identities_to_inference = []
        print(identities_to_inference)

        video_filename = 'HDTF_nvs_video.mp4'

        video_dst_dir = os.path.join(self.results_dst_dir, 'trajectory_videos')
        Path(video_dst_dir).mkdir(parents=True, exist_ok=True)

        writer = skvideo.io.FFmpegWriter(os.path.join(video_dst_dir,
                                                      video_filename),
                                         outputdict={
                                             '-pix_fmt': 'yuv420p',
                                         })

        # trajectory = self.create_trajectory(len(loader) * self.opt.inference.eval_batch)
        trajectory = self.create_trajectory(250)
        trajectory = trajectory.repeat(
            len(loader) * self.opt.inference.eval_batch // 250 + 1, 1)
        chunk = 1  # shared

        for j, batch in enumerate(tqdm(loader)):

            if j > 1500:
                break

            images = batch['image'].to(self.device)
            images = self.pool_256(images)

            ref_imgs_info = self.encode_ref_images(images)
            pred_latents = ref_imgs_info['pred_latents']

            que_latents = pred_latents  # ! no editing for now

            chunk_trajectory = trajectory[j:j +
                                          chunk]  # currently only 1 supported
            chunk_cam_settings = self._cam_locations_2_cam_settings(
                1, chunk_trajectory)

            que_info = self.latent2image(
                # pred_latents,
                que_latents,
                chunk_cam_settings,
                geometry_sample=None,  # just study pixel sup
                local_condition=None,
                sample_with_renderer=True,
                input_is_latent=True)  # add later

            que_render_ref_out = self.que_render_given_ref(
                ref_imgs_info, que_info)  # cross reconstruction

            res_render_out = que_render_ref_out['res_render_out']
            pred_imgs = self.pool_256(res_render_out['gen_imgs'])

            writer.writeFrame(Tensor2Array(pred_imgs))

            del res_render_out, que_info, que_render_ref_out, ref_imgs_info, images, pred_imgs, pred_latents
            torch.cuda.empty_cache()

        writer.close()
        self.on_val_end()
