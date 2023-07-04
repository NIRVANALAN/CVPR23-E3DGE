import os

import torch
from munch import Munch

# for autoencoder
from project.models.stylesdf_model import G_pred_latents as Generator


def setup_inference_configs(opt, device):  # todo
    opt.model.is_test = True
    opt.rendering.camera = opt.camera
    opt.model.freeze_renderer = False
    opt.rendering.offset_sampling = True
    opt.rendering.static_viewdirs = True
    opt.rendering.force_background = True
    opt.rendering.perturb = 0
    opt.inference.size = opt.model.size
    opt.inference.camera = opt.camera
    opt.inference.renderer_output_size = opt.model.renderer_spatial_output_dim
    opt.inference.style_dim = opt.model.style_dim
    opt.inference.project_noise = opt.model.project_noise
    opt.inference.return_xyz = opt.rendering.return_xyz

    # find checkpoint directory
    # check if there's a fully trained model
    checkpoints_dir = 'full_models'
    checkpoint_path = os.path.join(checkpoints_dir,
                                   opt.experiment.expname + '.pt')
    if os.path.isfile(checkpoint_path):
        # define results directory name
        result_model_dir = 'final_model'
    else:
        checkpoints_dir = os.path.join('checkpoint', opt.experiment.expname)
        checkpoint_path = os.path.join(
            checkpoints_dir,
            'models_{}.pt'.format(opt.experiment.ckpt.zfill(7)))
        # define results directory name
        result_model_dir = 'iter_{}'.format(opt.experiment.ckpt.zfill(7))

    # create results directory
    results_dir_basename = os.path.join(opt.inference.results_dir,
                                        opt.experiment.expname)
    opt.inference.results_dst_dir = os.path.join(results_dir_basename,
                                                 result_model_dir)
    if opt.inference.fixed_camera_angles:
        opt.inference.results_dst_dir = os.path.join(
            opt.inference.results_dst_dir, 'fixed_angles')
    else:
        opt.inference.results_dst_dir = os.path.join(
            opt.inference.results_dst_dir, 'random_angles')
    os.makedirs(opt.inference.results_dst_dir, exist_ok=True)
    os.makedirs(os.path.join(opt.inference.results_dst_dir, 'images'),
                exist_ok=True)
    if not opt.inference.no_surface_renderings:
        os.makedirs(os.path.join(opt.inference.results_dst_dir,
                                 'depth_map_meshes'),
                    exist_ok=True)
        os.makedirs(os.path.join(opt.inference.results_dst_dir,
                                 'marching_cubes_meshes'),
                    exist_ok=True)
        os.makedirs(os.path.join(opt.inference.results_dst_dir, 'meshes'),
                    exist_ok=True)
        os.makedirs(os.path.join(opt.inference.results_dst_dir,
                                 'shading_meshes'),
                    exist_ok=True)
        os.makedirs(os.path.join(opt.inference.results_dst_dir,
                                 'debug_meshes'),
                    exist_ok=True)

    # load saved model
    checkpoint = torch.load(checkpoint_path)

    # load image generation model
    g_ema = Generator(opt.model, opt.rendering).to(device)
    pretrained_weights_dict = checkpoint["g_ema"]
    model_dict = g_ema.state_dict()
    for k, v in pretrained_weights_dict.items():
        if v.size() == model_dict[k].size():
            model_dict[k] = v

    g_ema.load_state_dict(model_dict)

    # load a second volume renderer that extracts surfaces at 128x128x128 (or higher) for better surface resolution
    if not opt.inference.no_surface_renderings:
        opt['surf_extraction'] = Munch()
        opt.surf_extraction.rendering = opt.rendering
        opt.surf_extraction.model = opt.model.copy()

        opt.surf_extraction.rendering.N_samples = opt.surf_extraction.model.renderer_spatial_output_dim
        opt.surf_extraction.rendering.return_xyz = True
        opt.surf_extraction.rendering.return_sdf = True
        surface_g_ema = Generator(opt.surf_extraction.model,
                                  opt.surf_extraction.rendering,
                                  full_pipeline=False).to(device)

        # Load weights to surface extractor
        surface_extractor_dict = surface_g_ema.state_dict()
        for k, v in pretrained_weights_dict.items():
            if k in surface_extractor_dict.keys() and v.size(
            ) == surface_extractor_dict[k].size():
                surface_extractor_dict[k] = v

        surface_g_ema.load_state_dict(surface_extractor_dict)
    else:
        surface_g_ema = None

    # get the mean latent vector for g_ema
    if opt.inference.truncation_ratio < 1:
        with torch.no_grad():
            mean_latent = g_ema.mean_latent(opt.inference.truncation_mean,
                                            device)
    else:
        surface_mean_latent = None

    # get the mean latent vector for surface_g_ema
    if not opt.inference.no_surface_renderings:
        surface_mean_latent = mean_latent[0]
    else:
        surface_mean_latent = None

    # generate(opt.inference, g_ema, surface_g_ema, device, mean_latent,
    #          surface_mean_latent)
    return g_ema, surface_g_ema, mean_latent, surface_mean_latent
