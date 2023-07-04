# PTI: https://github.dev/danielroich/PTI
import project.utils.deca_util as util
import copy
import cv2
import gc
import glob
import json
import os
import traceback
from pathlib import Path

import numpy as np
import torch
from ipdb import set_trace as st
from PIL import Image
from project.data.now import NoWDataset
from project.utils import refresh_cache
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from tqdm import tqdm

from project.trainers.trainer import AERunner, RUNNER

# piGAN: https://github.com/marcoamonteiro/pi-GAN/blob/master/inverse_render.py
# SG2: https://github.com/NVlabs/stylegan2-ada/blob/main/projector.py


@RUNNER.register_module(force=True)
class Projectors(AERunner):
    """Implementation of SG2, SG2+, PTI inversions.
    """

    def __init__(
        self,
        encoder: torch.nn.Module,
        volume_discriminator: torch.nn.Module,
        generator: torch.nn.Module,
        mean_latent,
        opt: dict,
        # opt_training: dict,
        device: torch.device,
        loaders: dict,
        loss_class: torch.nn.Module = None,
        surface_g_ema: torch.nn.Module = None,
        ckpt=None,  # todo, to lint all params
        # experiment_opt: dict = None,
        mode='val',  # todo
        e_optim=None,
        work_dir=None,
        max_iters=None,
        max_epochs=None,
        discriminator=None,
    ):
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
        self.projection_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5),
                                 inplace=True)
        ])

    def _configure_optimizers(self, latent_in: list):
        # todo
        opt = self.opt.projection

        print(f'param_groups: {latent_in}')

        projection_optim = torch.optim.Adam(latent_in, lr=opt.first_inv_lr)
        self.projection_optim = projection_optim
        print(projection_optim)

    def _init_latent_code(self):
        opt = self.opt.projection

        if opt.wspace:
            latent_in_renderer = torch.zeros_like(
                self.mean_latent[0][:, 0, ...])  # optimise offset, B 9 256
            latent_in_decoder = torch.zeros_like(
                self.mean_latent[1][:, 1, ...])  # optimise offset, B 10 512
        else:  # wp space
            # init code
            latent_in_renderer = torch.zeros_like(
                self.mean_latent[0])  # optimise offset, B 9 256
            latent_in_decoder = torch.zeros_like(
                self.mean_latent[1])  # optimise offset, B 10 256

        latent_in_renderer.requires_grad_()
        latent_in_decoder.requires_grad_()
        return latent_in_renderer, latent_in_decoder

    def _preprocess_code_for_inverse(self, latent_in_offset: list, num_steps,
                                     step):
        opt = self.opt.projection
        latent_in_renderer, latent_in_decoder = latent_in_offset
        noise_latent_in_renderer = 0.03 * torch.randn_like(
            latent_in_renderer) * (num_steps - step) / num_steps
        noise_latent_in_decoder = 0.03 * torch.randn_like(
            latent_in_decoder) * (num_steps - step) / num_steps
        latent_in_offset = [
            noise_latent_in_renderer + latent_in_renderer,
            latent_in_decoder + noise_latent_in_decoder
        ]

        if opt.wspace:
            latent_in_offset = [
                latent.unsqueeze(1).repeat_interleave(style_num, 1)
                for latent, style_num in zip(latent_in_offset, (
                    mean_latent.shape[1] for mean_latent in self.mean_latent))
            ]

        latent_in = self._add_offset2latent(latent_in_offset, self.device)

        return latent_in

    def project(self, input_imgs=None, imgfile=None):
        opt = self.opt.projection
        assert input_imgs is None

        # st()

        if imgfile is not None:
            files_to_inverse = [imgfile]
        else:
            if len(opt.inverse_files) == 1:
                opt.inverse_files = sorted(glob.glob(opt.inverse_files[0]),
                                           reverse=opt.reverse_file_order)
            else:
                opt.inverse_files = sorted(opt.inverse_files,
                                           reverse=opt.reverse_file_order)
                files_to_inverse = opt.inverse_files

        self.eval_mode()
        # load imgs
        transform = self.projection_transforms

        inversed_latents_in = []
        imgs = []
        all_loss_dicts = []
        print(files_to_inverse)

        for imgfile in files_to_inverse:

            projection_imgsave_root_dir = Path(
                os.path.join(self.opt.results_dst_dir, 'projection',
                             Path(imgfile).stem))

            if (projection_imgsave_root_dir / 'latent_in.pt').exists():
                continue

            refresh_cache()
            if input_imgs is not None:
                img_name = Path(imgfile).name
                gt_img = input_imgs
            else:
                img_name = Path(imgfile).name
                gt_img = transform(Image.open(imgfile).convert("RGB"))
                gt_img = gt_img.unsqueeze(0).to(self.device)

            gt_img = self.pool_256(gt_img)
            gt_thumb_imgs = self.pool_64(gt_img)

            latent_in_renderer, latent_in_decoder = self._init_latent_code()
            latent_in_offset = [latent_in_renderer, latent_in_decoder]
            self._configure_optimizers(latent_in_offset)
            optimizer = self.projection_optim

            # optimization loop
            num_steps = opt.first_inv_steps
            pbar = range(num_steps + 1)
            pbar = tqdm(pbar, initial=1, dynamic_ncols=True, smoothing=0.01)
            # pred imgs cameras
            pred_cam_settings = self.image2camsettings(gt_thumb_imgs)

            projection_imgsave_root_dir.mkdir(parents=True, exist_ok=True)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                        100,
                                                        gamma=0.75)

            # https://github.com/marcoamonteiro/pi-GAN/blob/master/inverse_render.py
            for step in pbar:
                latent_in = self._preprocess_code_for_inverse(
                    latent_in_offset, num_steps, step)

                render_out = self.latent2image(
                    latent_in,
                    cam_settings=pred_cam_settings,
                    input_is_latent=True,
                    local_condition=None,
                )
                predictions = (self.pool_256(render_out['gen_imgs']),
                               render_out['gen_thumb_imgs'])

                pred = predictions[0]

                loss = torch.tensor(0.).to(self.device).float()
                lpips_loss = self.loss_module.criterionLPIPS(pred, gt_img)
                l2_loss = self.loss_module.criterionImg(pred, gt_img)
                loss += (l2_loss * opt.pt_l2_lambda +
                         lpips_loss * opt.pt_lpips_lambda)

                lpips_loss_thumb = self.loss_module.criterionLPIPS(
                    self.pool_256(predictions[1]), gt_img)
                l2_loss_thumb = self.loss_module.criterionImg(
                    self.pool_256(predictions[1]), gt_img)
                loss += (l2_loss_thumb * opt.pt_l2_lambda +
                         lpips_loss_thumb * opt.pt_lpips_lambda
                         ) * 0.1  # no loss on thumb leads to shape collapse

                loss.backward()
                if step != 0:
                    optimizer.step()
                    scheduler.step()
                optimizer.zero_grad()

                if step % opt.projection_logging_interval == 0 or step == opt.first_inv_steps - 1:
                    with torch.no_grad():
                        if not self.opt.inference.no_surface_renderings:
                            local_condition = None  # * don't add texture on geometry for now
                            surface_out = self.latent2surface(
                                latent_in,
                                True,
                                cam_settings=pred_cam_settings,
                                return_mesh=True,
                                local_condition=local_condition)

                            # also save as depth
                            self.save_depth_mesh(
                                surface_out,
                                cam_settings=pred_cam_settings,
                                mesh_saveprefix='meshes_{}'.format(step),
                                mesh_saveroot=str(projection_imgsave_root_dir))

                        img_save_path = projection_imgsave_root_dir / f'{step}.png'
                        utils.save_image(torch.cat([
                            self.pool_256(prediction)
                            for prediction in predictions
                        ] + [self.pool_256(gt_img)], -1),
                                         img_save_path,
                                         nrow=1,
                                         normalize=True,
                                         value_range=(-1, 1))
                        print('save at {}'.format(img_save_path))

                description = f"Iter: {step} imgname: {img_name}  l2-loss: {l2_loss.item():.4f} VGG-loss: {lpips_loss.item():.4f}"
                pbar.set_description((description))
                # gc.collect()
                # torch.cuda.empty_cache()
                refresh_cache()

            #=================== log metric =================
            _, loss_2d_rec_dict = self.loss_module.calc_2d_rec_loss(
                self.pool_256(render_out['gen_imgs']),
                gt_img,
                gt_img,
                self.opt.training,
                loss_dict=True,
                mode='val')  # 21.9G

            # loss_dict = {**loss_2d_rec_dict}
            all_loss_dicts.append(loss_2d_rec_dict)

            if self.opt.inference.save_independent_img:
                utils.save_image(self.pool_256(predictions[0]),
                                 Path(self.opt.results_dst_dir) /
                                 'projection' / img_name,
                                 nrow=1,
                                 normalize=True,
                                 value_range=(-1, 1))
            else:
                utils.save_image(torch.cat(
                    [self.pool_256(prediction)
                     for prediction in predictions] + [self.pool_256(gt_img)],
                    -1),
                                 Path(self.opt.results_dst_dir) /
                                 'projection' / img_name,
                                 nrow=1,
                                 normalize=True,
                                 value_range=(-1, 1))

            del loss_2d_rec_dict
            refresh_cache()

            latent_in_for_save = [
                latent.detach().cpu() for latent in latent_in
            ]
            self.on_project_end(latent_in_for_save,
                                projection_imgsave_root_dir)

            inversed_latents_in.append(latent_in_for_save)

        val_scores_for_logging = self._calc_average_loss(all_loss_dicts)
        with open(
                Path(self.opt.results_dst_dir) / 'projection' / 'scores.json',
                'a') as f:
            json.dump({**val_scores_for_logging, 'iter': self._iter}, f)

        # return inversed_latents_in
        return render_out

    def on_project_end(self, latent_in_for_save,
                       projection_imgsave_root_dir: str):
        torch.save({
            "latent_in": latent_in_for_save,
        }, projection_imgsave_root_dir / 'latent_in.pt')
        print('Successfully saved final latent at {}'.format(
            str(projection_imgsave_root_dir / 'latent_in.pt')))

    def run(self):
        if self.opt.inference.deca_eval:
            self.evaluate3D()
        else:
            return self.project()

    # @torch.no_grad()
    def evaluate3D(self, mode='val'):
        self.eval_mode()
        ''' NOW validation / test for optimization based mesh reconstruction.
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

            if Path(os.path.join(savefolder, f'{imagename[0]}.obj')).exists():
                continue

            aligned_rgb_Tensor_images = aligned_rgb_Tensor_images.to(
                self.device)

            assert len(imagename) == 1

            # === replace ====#
            name = imagename[0].split('/')[-1]
            img_file = os.path.join(savefolder, imagename[0], name + '.png')
            render_out = self.project(aligned_rgb_Tensor_images, img_file)

            if isinstance(render_out, tuple):  # * pti
                render_out, surface_out = render_out
            else:
                surface_out = None

            # === replace ====#
            with torch.no_grad():

                visdict = dict(input_imgs=aligned_rgb_Tensor_images)
                # get mesh & 3D LMS
                post_process_out = self.post_process_depthMesh(
                    aligned_cv2_img, render_out, surface_out)
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
        self.train_mode()


@RUNNER.register_module(force=True)
class ProjectorsPTI(Projectors):
    """Implementation of SG2, SG2+, PTI inversions.
    """

    def __init__(
        self,
        encoder: torch.nn.Module,
        volume_discriminator: torch.nn.Module,
        generator: torch.nn.Module,
        mean_latent,
        opt: dict,
        # opt_training: dict,
        device: torch.device,
        loaders: dict,
        loss_class: torch.nn.Module = None,
        surface_g_ema: torch.nn.Module = None,
        ckpt=None,  # todo, to lint all params
        # experiment_opt: dict = None,
        mode='val',  # todo
        e_optim=None,
        work_dir=None,
        max_iters=None,
        max_epochs=None,
        discriminator=None,
    ):
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

    def _configure_optimizers(self, g_module_pti):
        # todo
        opt = self.opt.projection
        projection_optim = torch.optim.Adam([
            {
                'name': 'G',
                'params': g_module_pti.parameters(),
                'lr': opt.pti_learning_rate
            },
        ])
        # print(f'param_groups: {projection_optim}')
        self.projection_optim = projection_optim

    def _init_latent_code(self, img_stem, input_imgs, imgfile):
        opt = self.opt.projection

        inversed_w_latent_in = Path(
            opt.w_inversion_root) / img_stem / 'latent_in.pt'
        # st()

        if inversed_w_latent_in.exists():
            latent_in = torch.load(inversed_w_latent_in)['latent_in']
            latent_in = [latent.to(self.device) for latent in latent_in]
        else:
            return None, None
            latent_in = super().project(input_imgs, imgfile)[0]['pred_latents']

        latent_in_renderer, latent_in_decoder = latent_in

        return latent_in_renderer, latent_in_decoder

    def project(self, input_imgs=None, imgfile=None):
        opt = self.opt.projection
        torch.manual_seed(0)
        transform = self.projection_transforms
        self.eval_mode()
        # * load pre-inversed code.

        inversed_latents_in = []
        all_loss_dicts = []
        g_module_pti = copy.deepcopy(self.g_module)
        surface_pti_generator = copy.deepcopy(self.surface_g_ema)

        if imgfile is not None:
            files_to_inverse = [imgfile]
        else:
            if len(opt.inverse_files) == 1:
                opt.inverse_files = sorted(glob.glob(opt.inverse_files[0]),
                                           reverse=opt.reverse_file_order)
            else:
                opt.inverse_files = sorted(opt.inverse_files,
                                           reverse=opt.reverse_file_order)
                files_to_inverse = opt.inverse_files

        # load imgs
        inversed_latents_in = []
        all_loss_dicts = []
        files_to_inverse.reverse()
        print(files_to_inverse)

        for imgfile in files_to_inverse:
            refresh_cache()
            img_stem = Path(imgfile).stem
            if input_imgs is not None:
                img_name = Path(imgfile).name
                gt_img = input_imgs
            else:
                img_name = Path(imgfile).name
                gt_img = transform(Image.open(imgfile).convert("RGB"))
                gt_img = gt_img.unsqueeze(0).to(self.device)

            gt_img = self.pool_256(gt_img)
            gt_thumb_imgs = self.pool_64(gt_img)

            projection_imgsave_root_dir = Path(
                os.path.join(self.opt.results_dst_dir, 'pti_projection',
                             Path(imgfile).stem))
            projection_imgsave_root_dir.mkdir(parents=True, exist_ok=True)
            if (projection_imgsave_root_dir / '350.png').exists():
                continue

            g_module_pti.load_state_dict(self.g_module.state_dict())

            latent_in_renderer, latent_in_decoder = self._init_latent_code(
                img_stem, input_imgs, imgfile)
            if latent_in_renderer is None:
                continue
            latent_in = [latent_in_renderer, latent_in_decoder
                         ]  # * here, already inversed add offsets
            self._configure_optimizers(g_module_pti)
            optimizer = self.projection_optim

            # optimization loop
            num_steps = opt.max_pti_steps
            pbar = range(num_steps + 1)
            pbar = tqdm(pbar, initial=1, dynamic_ncols=True, smoothing=0.01)
            # pred imgs cameras
            pred_cam_settings = self.image2camsettings(gt_thumb_imgs)

            # https://github.com/marcoamonteiro/pi-GAN/blob/master/inverse_render.py
            for step in pbar:

                render_out = self.latent2image(latent_in,
                                               cam_settings=pred_cam_settings,
                                               input_is_latent=True,
                                               local_condition=None,
                                               pti_generator=g_module_pti
                                               # pti_generator=None
                                               )
                predictions = (self.pool_256(render_out['gen_imgs']),
                               render_out['gen_thumb_imgs'])
                pred = predictions[0]
                assert pred.ndim == 4  # ! for lpips loss, must has batch dimention

                loss = torch.tensor(0.).to(self.device).float()

                # for pred, gt in zip((predictions[0]), (input_imgs)): # inaccurate lpips loss for thumb imgs
                lpips_loss = self.loss_module.criterionLPIPS(pred, input_imgs)
                l2_loss = self.loss_module.criterionImg(pred, input_imgs)
                loss += (l2_loss * opt.pt_l2_lambda +
                         lpips_loss * opt.pt_lpips_lambda)

                lpips_loss_thumb = self.loss_module.criterionLPIPS(
                    self.pool_256(predictions[1]), input_imgs)
                l2_loss_thumb = self.loss_module.criterionImg(
                    self.pool_256(predictions[1]), input_imgs)
                loss += (l2_loss_thumb * opt.pt_l2_lambda +
                         lpips_loss_thumb * opt.pt_lpips_lambda
                         ) * 0.1  # no loss on thumb leads to shape collapse

                try:
                    if step != 0:
                        loss.backward()
                        optimizer.step()
                except Exception as e:
                    st()

                optimizer.zero_grad()

                if step % 50 == 0 or step == opt.first_inv_steps - 1:
                    with torch.no_grad():
                        if not self.opt.inference.no_surface_renderings:
                            # st()
                            surface_pti_generator.load_state_dict(
                                g_module_pti.state_dict(),
                                strict=False)  # no decoder here
                            local_condition = None  # * don't add texture on geometry for now
                            try:
                                torch.cuda.empty_cache()
                                surface_out = self.latent2surface(
                                    latent_in,
                                    True,
                                    cam_settings=pred_cam_settings,
                                    return_mesh=False,
                                    local_condition=local_condition,
                                    pti_generator=surface_pti_generator)

                                # also save as depth
                                self.save_depth_mesh(
                                    surface_out,
                                    cam_settings=pred_cam_settings,
                                    mesh_saveprefix='meshes_{}'.format(step),
                                    mesh_saveroot=str(
                                        projection_imgsave_root_dir))
                            except Exception as e:
                                traceback.print_exc()
                                torch.cuda.empty_cache()

                        img_save_path = projection_imgsave_root_dir / f'{step}.png'
                        utils.save_image(torch.cat([
                            self.pool_256(prediction)
                            for prediction in predictions
                        ] + [self.pool_256(input_imgs)], -1),
                                         img_save_path,
                                         nrow=1,
                                         normalize=True,
                                         value_range=(-1, 1))
                        print('save at {}'.format(img_save_path))

                description = f"Iter: {step} imgname: {img_name}  l2-loss: {l2_loss.item():.4f} VGG-loss: {lpips_loss.item():.4f} ThumbL2: {l2_loss_thumb.item():.4f} VGG-thumb-loss: {lpips_loss_thumb.item():.4f}"
                pbar.set_description((description))

            #=================== log metric =================
            _, loss_2d_rec_dict = self.loss_module.calc_2d_rec_loss(
                pred,
                input_imgs,
                input_imgs,
                self.opt.training,
                loss_dict=True,
                mode='val')  # 21.9G

            # loss_dict = {**loss_2d_rec_dict}
            all_loss_dicts.append(loss_2d_rec_dict)

            if self.opt.inference.save_independent_img:
                utils.save_image(self.pool_256(predictions[0]),
                                 Path(self.opt.results_dst_dir) /
                                 'pti_projection' / img_name,
                                 nrow=1,
                                 normalize=True,
                                 value_range=(-1, 1))

            else:
                utils.save_image(torch.cat(
                    [self.pool_256(prediction) for prediction in predictions] +
                    [self.pool_256(input_imgs)], -1),
                                 Path(self.opt.results_dst_dir) /
                                 'pti_projection' / img_name,
                                 nrow=1,
                                 normalize=True,
                                 value_range=(-1, 1))

            latent_in_for_save = [
                latent.detach().cpu() for latent in latent_in
            ]
            self.on_project_end(latent_in_for_save,
                                projection_imgsave_root_dir, g_module_pti)
            inversed_latents_in.append(latent_in_for_save)

            del latent_in_for_save, predictions
            if not self.opt.inference.deca_eval and not self.opt.inference.no_surface_renderings:
                del surface_out

            gc.collect()
            torch.cuda.empty_cache()

        # * todo
        val_scores_for_logging = self._calc_average_loss(all_loss_dicts)
        with open(
                Path(self.opt.results_dst_dir) / 'pti_projection' /
                'scores.json', 'a') as f:
            json.dump({**val_scores_for_logging, 'iter': self._iter}, f)

        return render_out, surface_out

    def on_project_end(self, latent_in_for_save,
                       projection_imgsave_root_dir: str, g_module_pti):
        if not self.opt.projection.inversion_calc_metrics:
            torch.save(
                {
                    "latent_in": latent_in_for_save,
                    "g": g_module_pti.state_dict()
                }, projection_imgsave_root_dir / 'latent_in.pt')
            print('Successfully saved final latent and G weights at {}'.format(
                str(projection_imgsave_root_dir / 'latent_in.pt')))
        else:
            print('ignore latent in saving')
