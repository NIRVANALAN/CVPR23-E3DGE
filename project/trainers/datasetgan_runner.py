import facexlib.utils.face_restoration_helper as face_restoration_helper
import mmcv  # to replace cv2 and plt
import torch
# from project.apis.sampling import generate
from project.utils import (DATASETGAN_3D, mixing_noise, sample_data)
from torchvision import transforms

from .base_runner import RUNNER, BaseRunner

try:
    import wandb
except ImportError:
    wandb = None


@RUNNER.register_module(force=True)
class DatasetGANRunner(BaseRunner):
    """A runner that could use GAN sampled data as training corpus.

    Args:
        BaseRunner (_type_): _description_
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
        # e_optim=None,
        work_dir=None,
        max_iters=None,
        max_epochs=None,
        discriminator=None,
    ):
        super().__init__(encoder, opt, work_dir, max_iters, max_epochs)
        self.volume_discriminator = volume_discriminator

        self.generator = generator
        self.discriminator = discriminator
        self.device = device
        self.mean_latent = mean_latent
        self.w_mean_latent = [
            latent[:, 0, :] if latent is not None else latent
            for latent in mean_latent
        ]
        self.pool_64 = torch.nn.AdaptiveAvgPool2d((64, 64)) # type: ignore
        self.pool_256 = torch.nn.AdaptiveAvgPool2d((256, 256)) # type: ignore

        self.loss_class = loss_class
        self.surface_g_ema = surface_g_ema
        self.train_loader = sample_data(loaders['train'])
        self.eval_loader, self.test_loader = (loaders[k]
                                              for k in ('val', 'test'))
        self.loaders = loaders
        # self._iter = 0
        opt = self.train_opt
        self.FaceRestoreHelper = face_restoration_helper.FaceRestoreHelper(
            upscale_factor=1, face_size=256)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5),
                                 inplace=True)
        ])

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

        # todo, extra network weights here.
        # * freeze parameters before calling DDP

        # * shared flags
        self.synthetic_data_flag = False
        self.enable_eikonal = self.train_opt.eikonal_lambda > 0 or self.train_opt.return_surface_eikonal > 0

    def on_train_step_start(self, *args):
        opt_training = self.train_opt
        # * CREATE dataset
        with torch.no_grad():
            self.GANSampleDataset = DATASETGAN_3D(
                self.g_module, self.g_module.mean_latent(10000, self.device),
                opt_training, opt_training.synthetic_batch_size)

            self.sample_noise = [
                torch.randn(opt_training.val_n_sample,
                            opt_training.style_dim,
                            device=self.device).repeat_interleave(8, dim=0)
            ]

    def synthetic_data_sample(self, noise=None, same_view=False):
        opt_training = self.opt.training
        GANSampleDataset = self.GANSampleDataset
        self.synthetic_data_flag = True

        if noise is None:
            if self.fixed_latents is None:
                noise = mixing_noise(opt_training.synthetic_batch_size,
                                     opt_training.style_dim,
                                     opt_training.mixing,
                                     self.device)  # * todo
            else:
                noise = [
                    self.fixed_latents[self._iter:self._iter + 1].to(
                        self.device)
                ]
        else:
            noise = [n.detach() for n in noise]

        # sample synthetic data
        random_3d_sample_batch, rand_cam_settings = GANSampleDataset.sample_with_rand_cams(
            noise,
            self.device,
            return_cam=True,
            sample_with_decoder=opt_training.full_pipeline,
            return_surface_eikonal=opt_training.return_surface_eikonal,
            return_eikonal=opt_training.eikonal_lambda > 0
            or opt_training.return_surface_eikonal,
            truncation=opt_training.truncation_ratio,
            sample_without_grad=True,
            iter_idx=self._iter,
            same_view=same_view)  # increase sampling photorealism

        random_3d_sample_batch.update(dict(noise=noise))

        if opt_training.full_pipeline:
            curr_fake_imgs = self.pool_256(
                random_3d_sample_batch['gen_imgs']
            )  # resize to 256 for encoder training
        else:
            curr_fake_imgs = random_3d_sample_batch['gen_thumb_imgs']

        return dict(
            fake_imgs=curr_fake_imgs,  # 256 resolution by default
            sample_batch=random_3d_sample_batch,
            cam_settings=rand_cam_settings)
