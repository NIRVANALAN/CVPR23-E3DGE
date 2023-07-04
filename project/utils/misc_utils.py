import gc
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from munch import Munch
from omegaconf.dictconfig import DictConfig

import cv2
import IPython.display
import numpy as np
import PIL.Image

from .camera_utils import generate_camera_params

# import numpy as np
gt_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
thumb_pool = torch.nn.AdaptiveAvgPool2d((64, 64))


def vis_tensor(tensor_img: torch.Tensor, normalize=False):
    if tensor_img.ndim == 4:
        tensor_img = tensor_img[0]  # only show the first dim
    np_img = tensor_img.detach().permute(1, 2, 0).cpu().numpy()
    if normalize:
        np_img = (np_img / 2) + 0.5


#     img = Image.fromarray(np_img)
    return np_img


def plt_2_cv2(np_img, bgr=True):
    assert np_img.min() >= 0 and np_img.max() <= 1
    if bgr:
        np_img = np_img[..., ::-1]
    np_img *= 255
    # return (np_img*255).astype(np.uint8)
    return np_img.astype(np.uint8)


def normalize_depth_img(depth_img: torch.Tensor):
    depth_min = depth_img.min()
    return (depth_img.clone() - depth_min) / depth_img.max()


def torch_to_numpy(x):
    assert isinstance(x, torch.Tensor)
    return x.detach().cpu().numpy()


############################## Model weights util functions #################
# Turn model gradients on/off
def requires_grad(model: torch.nn.Module, flag=True):
    if model is None:
        return
    for p in model.parameters():
        p.requires_grad = flag


def generate_sample(GANSampleDataset, opt, device):
    sample_noise = [
        torch.randn(opt.val_n_sample, opt.style_dim,
                    device=device).repeat_interleave(8, dim=0)
    ]
    sample_cam_settings = generate_camera_params(
        opt.renderer_output_size,
        device,
        batch=opt.val_n_sample,
        sweep=True,
        uniform=opt.camera.uniform,
        azim_range=opt.camera.azim,
        elev_range=opt.camera.elev,
        fov_ang=opt.camera.fov,
        dist_radius=opt.camera.dist_radius,
        return_calibs=True)

    sample_cam_poses, sample_focals, sample_near, sample_far, sample_locations = [
        sample_cam_settings[key]
        for key in ['poses', 'focal', 'near', 'far', 'viewpoint']
    ]

    for j in range(0, opt.val_n_sample * 8, opt.chunk):
        fixed_3d_sample_data = GANSampleDataset.sample_shared(
            [noise[j:j + opt.chunk] for noise in sample_noise],
            {
                k: v[j:j + opt.chunk]
                for k, v in sample_cam_settings.items()
            },
            # sample_with_decoder=True,
            return_eikonal=False,
            return_surface_eikonal=False)  # todo, add flag

    return fixed_3d_sample_data, sample_noise, sample_cam_settings


def refresh_cache():
    torch.cuda.empty_cache()
    gc.collect()


def remove_module(state_dict):
    # create new OrderedDict that does not contain `module.`
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict


def imshow(images, col, viz_size=256):
    """Shows images in one figure."""
    import io

    if isinstance(images, torch.Tensor):
        if images.ndim == 4 and images.shape[1] == 3:
            images = images.permute(0, 2, 3, 1)  #  nhwc
        images = images.detach().cpu().numpy()
        images = images / 2 + 0.5
        # here use uint8, so normalize back to 255
        images = images * 255

    num, height, width, channels = images.shape
    assert num % col == 0
    row = num // col

    fused_image = np.zeros((viz_size * row, viz_size * col, channels),
                           dtype=np.uint8)

    for idx, image in enumerate(images):
        i, j = divmod(idx, col)
        y = i * viz_size
        x = j * viz_size
        if height != viz_size or width != viz_size:
            image = cv2.resize(image, (viz_size, viz_size))
        fused_image[y:y + viz_size, x:x + viz_size] = image

    fused_image = np.asarray(fused_image, dtype=np.uint8)
    data = io.BytesIO()
    PIL.Image.fromarray(fused_image).save(data, 'jpeg')
    im_data = data.getvalue()
    disp = IPython.display.display(IPython.display.Image(im_data))
    return disp, fused_image


class PosEncoding(nn.Module):

    def __init__(self, in_channels, N_freqs, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
        # super(PosEncoding, self).__init__()
        super().__init__()
        self.N_freqs = N_freqs
        self.in_channels = in_channels
        self.funcs = [torch.sin, torch.cos]
        self.out_channels = in_channels * (len(self.funcs) * N_freqs + 1)

        if logscale:
            self.freq_bands = 2**torch.linspace(0, N_freqs - 1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2**(N_freqs - 1), N_freqs)

    def forward(self, x):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...) 
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12

        Inputs:
            x: (B, self.in_channels)

        Outputs:
            out: (B, self.out_channels)
        """
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq * x)]

        return torch.cat(out, -1)


def Tensor2Array(images):
    if images.ndim == 4 and images.shape[1] == 3:
        images = images.permute(0, 2, 3, 1)  #  nhwc
    images = images.detach().cpu().numpy()
    images = images / 2 + 0.5
    # here use uint8, so normalize back to 255
    images = images * 255
    return images


def __get_keys(d, name):  # * extracts state_dicts from class-based module
    if 'state_dict' in d:
        d = d['state_dict']
    d_filt = {
        k[len(name) + 1:]: v
        for k, v in d.items() if k[:len(name)] == name
    }
    return d_filt


def load_state_dict_match_size(ckpt, model_state_dict):

    for k, v in ckpt.items():
        if k in model_state_dict.keys() and v.size(
        ) == model_state_dict[k].size():
            model_state_dict[k] = v
    return model_state_dict


def get_trainable_parameter(module):
    trainable_params = []
    for name, param in module.named_parameters():
        if param.requires_grad:
            trainable_params.append(param)
    return trainable_params
    # return len(trainable_params)>0, (param for parm in trainable_params) # convert to generator


def print_parameter(module):
    for name, param in module.named_parameters():
        if param.requires_grad:
            print(name, param.shape, flush=True)


def dictconfig_to_munch(d):
    """Convert object of type OmegaConf to Munch so Wandb can log properly
    Support nested dictionary.
    """
    return Munch({
        k: dictconfig_to_munch(v) if isinstance(v, DictConfig) else v
        for k, v in d.items()
    })


def scandir(dir_path, suffix=None, recursive=False, full_path=False):
    """Scan a directory to find the interested files.
    https://github.dev/XPixelGroup/BasicSR

    Args:
        dir_path (str): Path of the directory.
        suffix (str | tuple(str), optional): File suffix that we are
            interested in. Default: None.
        recursive (bool, optional): If set to True, recursively scan the
            directory. Default: False.
        full_path (bool, optional): If set to True, include the dir_path.
            Default: False.

    Returns:
        A generator for all the interested files with relative paths.
    """

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                if full_path:
                    return_path = entry.path
                else:
                    return_path = os.path.relpath(entry.path, root)

                if suffix is None:
                    yield return_path
                elif return_path.endswith(suffix):
                    yield return_path
            else:
                if recursive:
                    yield from _scandir(entry.path,
                                        suffix=suffix,
                                        recursive=recursive)
                else:
                    continue

    return _scandir(dir_path, suffix=suffix, recursive=recursive)


def crop_one_img(FaceRestoreHelper, img, save_cropped_path=None):
    """input: cv2 img; 
    output aligned cv2 img
    """
    FaceRestoreHelper.clean_all()
    FaceRestoreHelper.read_image(img)
    # get face landmarks
    FaceRestoreHelper.get_face_landmarks_5()
    return FaceRestoreHelper.align_warp_face(save_cropped_path)


def landmark_98_to_7(landmark_98):
    """Transfer 98 landmark positions to NoW evaluation 2D landmark positions.
    Args:
        landmark_98(numpy array): Polar coordinates of 98 landmarks, (98, 2)
    Returns:
        landmark_7(numpy array): Polar coordinates of 7 landmarks, (7, 2)
    """
    # 60 64 68 72 57 76-78 92-98

    landmark_7 = np.zeros((7, 2), dtype='float32')

    # eyes, left 2 right 2
    landmark_7[0, :] = landmark_98[60, :]
    landmark_7[1, :] = landmark_98[64, :]
    landmark_7[2, :] = landmark_98[68, :]
    landmark_7[3, :] = landmark_98[72, :]
    # nose
    landmark_7[4, :] = landmark_98[57, :]
    # left-right mouse
    landmark_7[5, :] = landmark_98[76, :]
    landmark_7[6, :] = landmark_98[92, :]

    return landmark_7
