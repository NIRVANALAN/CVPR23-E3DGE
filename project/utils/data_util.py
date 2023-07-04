from torch.utils import data
from dataclasses import dataclass
import numpy as np
import torch
from munch import Munch

from .camera_utils import generate_camera_params
from .training_utils import mixing_noise


######################### Dataset util functions ###########################
# Get data sampler
def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


# Get data minibatch
def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


@dataclass
class DATASETGAN_3D:
    gan_module: torch.nn.Module
    mean_latent: list
    opt: Munch
    batch_size: int
    latents: torch.Tensor = None  # randn(SIZE, 256) for training or testing
    random_sample: bool = True  # when true, sample from randn with given batch_size
    B_MIN: np.array = np.array([-0.15, -0.15, -0.15])
    B_MAX: np.array = np.array([0.15, 0.15, 0.15])
    thumb_pool: torch.nn.Module = torch.nn.AdaptiveAvgPool2d((64, 64))
    gt_pool: torch.nn.Module = torch.nn.AdaptiveAvgPool2d((256, 256))
    iter_idx: int = 0

    # curriculum_pose_sampling: int = -1

    # sample

    def __len__(self):
        if self.random_sample:
            return self.latents.shape[0]
        return -1  # random sample

    def sample_shared(self,
                      noise: list,
                      camera_settings: dict,
                      sample_with_decoder: bool = False,
                      return_eikonal: bool = False,
                      return_surface_eikonal: bool = True,
                      **kwargs):
        """sample shape and rendering from z space given noise and camera settings.
        shared & merge sampleing call fn. Sample images & geometry information given randn and camera-settings
        """
        # self.current_iter += 1
        opt = self.opt  # opt_training
        cam_poses, focal, near, far = [
            camera_settings[key] for key in ['poses', 'focal', 'near', 'far']
        ]

        sample_batch = []
        # st() # check: sample_with_decoder
        for j in range(0, noise[0].shape[0], opt.chunk):
            curr_noise = [n[j:j + opt.chunk] for n in noise]
            curr_batch = self.gan_module(
                curr_noise,
                cam_poses[j:j + opt.chunk],
                focal[j:j + opt.chunk],
                near[j:j + opt.chunk],
                far[j:j + opt.chunk],
                truncation_latent=self.mean_latent,
                # truncation=opt.truncation_ratio,
                sample_with_renderer=not sample_with_decoder,
                input_is_latent=False,
                sample_mode=True,
                return_xyz=True,
                return_eikonal=return_eikonal,
                # return_mesh=True,
                # mesh_with_shading=True,
                return_surface_eikonal=return_surface_eikonal,
                sample_with_decoder=sample_with_decoder,  # Stupid LYS.
                **kwargs)  # Todo

            # st() # check mesh and shading in cur_batch dict
            sample_batch.append(curr_batch)

        sample_batch = self.merge_sampled_batch(sample_batch)
        # st() # check mesh and shading in dict
        # check gan module output latent: whether mapped or not? the render_surface mapped again

        # resize 1024 sampling to 256
        # if sample_with_decoder:
        #     sample_batch['gen_imgs'] = self.gt_pool(sample_batch['gen_imgs'])

        return {**sample_batch, **camera_settings}

    def sample_surface_3d_data(self):
        """for sample geometry data 3D points, sdf, normal etc on surface.
        """

    def sample_uniform_surface(self):
        """uniform sample 3d points around surface.
        """

    def random_perm_and_shuffle(self, data_batch, batch_size):
        """for points, shuffle and return certain batch size

        Args:
            data_batch: returned from sample_batch_shared()
            batch_size: >B*64*64
        """
        # random shuffle & sampling first batch_size # of pts

        # add randn to surface
        # add rand to uniform points

        surface_pts_num = data_batch['xyz'].shape(-1)
        uniform_pts_num = data_batch[''].shape(-1)

        surf_rand_perm_idx = torch.randperm(surface_pts_num)
        uniform_pts_perm_idx = torch.randperm(surface_pts_num)

        # permute surface points
        for k in ['xyz', 'surface_eikonal_term']:
            pass

        for k in ['points', 'sdf']:
            pass

    def sample_with_rand_cams(self,
                              noise: list,
                              device,
                              return_cam: bool = False,
                              sample_with_decoder: bool = False,
                              iter_idx=-1,
                              same_view=False,
                              **kwargs):
        """randomly sample image batch given randn or predefined batch_size. Used in training image-branch

        Args:
            rand_n (torch.Tensor): random noise
        """
        self._iter = max(0, iter_idx)

        # with torch.no_grad():
        rand_camera_settings = self.sample_camera_poses(
            device, batch=noise[0].shape[0])

        # G(z) -> Imgs
        # gen_imgs = []
        # geometry_data = []
        sample_batch = self.sample_shared(noise, rand_camera_settings,
                                          sample_with_decoder, **kwargs)
        if return_cam:
            return sample_batch, rand_camera_settings

        return sample_batch

        # return {**sample_batch, **rand_camera_settings}

    def merge_sampled_batch(self, sample_batch):
        # KEYS = ('rgb_map', 'eikonal_term', 'xyz', 'sdf', 'mask', 'calibs', 'surface_eikonal_term', 'points')
        # todo, merge
        KEYS = ('gen_thumb_imgs', 'gen_imgs', 'xyz', 'sdf', 'mask', 'calibs',
                'surface_eikonal_term', 'points', 'mesh', 'shading_mesh')

        data_batch = sample_batch[0]

        for idx in range(1, len(sample_batch)):
            for k, v in sample_batch[idx].items(
            ):  # didn't check in manual merge
                if v is None or k not in KEYS:
                    continue
                data_batch[k] = torch.cat([data_batch[k], v],
                                          0)  # to optimize?
                # print(data_batch[k].shape)

        return data_batch

    def sample_sdf_surface(self, ray_marching=False):
        # todo, sample surface points via ray-marching
        pass

    def get_curriculum_pose_lambda(self, current_step):
        opt = self.opt  # training opt
        if opt.progressive_pose_sampling:
            progressive_pose_lambda = opt.progressive_pose_lambda
            progressive_pose_steps = opt.progressive_pose_steps
            assert len(progressive_pose_steps) == len(progressive_pose_lambda)

            for progressive_interval in range(len(progressive_pose_steps)):
                if not current_step >= progressive_pose_steps[
                        progressive_interval]:
                    break

            if progressive_interval != len(progressive_pose_steps):
                progressive_interval -= 1

            current_pose_lambda = progressive_pose_lambda[progressive_interval]

            return current_pose_lambda

        return 1

    def sample_camera_poses(self, device, batch=None, predefined_cam_opt=None):
        if predefined_cam_opt is None:
            opt = self.opt
        else:
            opt = predefined_cam_opt  # etc, trajectory

        if opt.progressive_pose_sampling > 0:
            progressive_pose_lambda = self.get_curriculum_pose_lambda(
                self._iter)
            # min(max(0, self._iter / opt.curriculum_pose_sampling_iternum), 1)

            azim_range = opt.camera.azim * progressive_pose_lambda
            elev_range = opt.camera.elev * progressive_pose_lambda
        else:
            azim_range = opt.camera.azim
            elev_range = opt.camera.elev

        camera_settings = generate_camera_params(  # * dict, with camera settings
            opt.renderer_output_size,
            device,
            batch=opt.batch if batch is None else batch,
            uniform=opt.camera.uniform,
            azim_range=azim_range,
            elev_range=elev_range,
            fov_ang=opt.camera.fov,
            dist_radius=opt.camera.dist_radius,
            azim_mean=opt.camera.azim_mean,
            elev_mean=opt.camera.elev_mean,
            return_calibs=True)
        return camera_settings

    def __getitem__(self, device, key):
        opt = self.opt

        noise = mixing_noise(opt.batch, opt.style_dim, opt.mixing, self.device)

        camera_settings = self.sample_camera_poses()

        cam_poses, focal, near, far, gt_viewpoint = [
            camera_settings[key]
            for key in ['poses', 'focal', 'near', 'far', 'viewpoint']
        ]

        # G(z) -> SDF, grad_sdf, 3D Points
        geometry_data_batch = self.GAN_module.data_sample_forward(
            noise, cam_poses, focal, near, far)
