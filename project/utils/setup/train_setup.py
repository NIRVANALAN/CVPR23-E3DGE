import gc
import json

import wandb
# from torch.utils.tensorboard import SummaryWriter

from pathlib import Path

import torch
from munch import Munch
from project.losses.builder import LossClass
from project.data.dataset import (ImagesDatasetEval, MultiResolutionDatasetLMS)
from project.models.encoders import set_encoder
# for autoencoder
from project.models.stylesdf_model import Discriminator
from project.models.stylesdf_model import G_pred_latents as Generator
from project.models.stylesdf_model import VolumeRenderDiscriminator
from project.models.op import grid_sample_gradfix
from project.utils import data_sampler
from project.utils.dist_utils import get_rank, synchronize
from torch.utils import data
from torchvision import transforms

from ..misc_utils import __get_keys, load_state_dict_match_size


def setup_training_configs(opt, device='cuda', return_loss_class_only=False):
    opt.training.camera = opt.camera
    opt.rendering.camera = opt.camera
    opt.training.size = opt.model.size
    opt.training.renderer_output_size = opt.model.renderer_spatial_output_dim
    opt.training.style_dim = opt.model.style_dim
    opt.model.freeze_renderer = True

    opt.training.offset_sampling = True
    # opt.training.static_viewdirs = True
    opt.training.force_background = True
    # opt.training.perturb = 0
    opt.training.size = opt.model.size
    opt.training.camera = opt.camera
    opt.training.renderer_output_size = opt.model.renderer_spatial_output_dim
    opt.training.style_dim = opt.model.style_dim
    opt.training.project_noise = opt.model.project_noise
    opt.training.channel_multiplier = opt.model.channel_multiplier
    opt.training.return_xyz = opt.rendering.return_xyz
    opt.training.dataset_path = opt.dataset.dataset_path
    opt.training.truncation_ratio = opt.inference.truncation_ratio

    opt.training.eval_dataset_path = opt.dataset.eval_dataset_path
    opt.training.save_img = opt.inference.save_img
    opt.training.enable_local_model = opt.rendering.enable_local_model

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

    if opt.training.enable_custom_grid_sample:
        grid_sample_gradfix.enabled = True

    map_location = {
        'cuda:%d' % 0: 'cuda:%d' % opt.training.local_rank
    }  # configure map_location properly

    import os
    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    opt.training.distributed = n_gpu > 1

    if opt.training.distributed:
        torch.cuda.set_device(opt.training.local_rank)
        print(opt.training.local_rank, flush=True)
        torch.distributed.init_process_group(backend="nccl",
                                             init_method="env://")
        synchronize()
    torch.cuda.empty_cache()

    exp_root = Path(
        os.path.join(opt.training.checkpoints_dir, opt.experiment.expname))
    #  mode))
    opt.results_dst_dir = str(exp_root)

    for mode in ['val', 'train']:
        mode_exp_dir = exp_root / mode
        mode_exp_dir.mkdir(parents=True, exist_ok=True)
        (mode_exp_dir / 'meshes').mkdir(parents=True, exist_ok=True)
        (mode_exp_dir / 'images').mkdir(parents=True, exist_ok=True)
        (mode_exp_dir / 'images').mkdir(parents=True, exist_ok=True)

    # torch.cuda.set_device(get_rank())

    # todo, load ckpt
    generator = Generator(opt.model,
                          opt.rendering,
                          full_pipeline=opt.training.full_pipeline).to(device)
    # g_ema = Generator(opt.model,
    #                   opt.rendering,
    #                   ema=True,
    #                   full_pipeline=opt.training.full_pipeline).to(device)
    # g_ema.eval()

    encoder = set_encoder(opt).to(device)
    volume_discriminator = VolumeRenderDiscriminator(
        opt.model).to(device).eval()

    # ================== load surface_g_ema if query surface (deprecated?) ==================
    if not opt.inference.no_surface_renderings:
        opt['surf_extraction'] = Munch()
        opt.surf_extraction.rendering = opt.rendering
        # opt.surf_extraction.rendering.spatial_super_sampling_factor = 4 # hard coded, or it will affect the SR module performance.
        opt.surf_extraction.rendering.spatial_super_sampling_factor = 1  # hard coded, or it will affect the SR module performance.
        opt.surf_extraction.model = opt.model.copy()
        opt.surf_extraction.model.renderer_spatial_output_dim = 128
        opt.surf_extraction.rendering.N_samples = opt.surf_extraction.model.renderer_spatial_output_dim
        opt.inference.surf_extraction_output_size = opt.surf_extraction.model.renderer_spatial_output_dim
        opt.surf_extraction.rendering.return_xyz = True
        opt.surf_extraction.rendering.return_sdf = True

        surface_g_ema = Generator(opt.surf_extraction.model,
                                  opt.surf_extraction.rendering,
                                  full_pipeline=False).to(device)
    else:
        surface_g_ema = None

    # ===========DEFINE LOSS =============
    # st()
    loss_class = LossClass(device, opt.training).to(device)
    if return_loss_class_only:
        return loss_class

    # * load pretrained checkpoints

    volume_discriminator_ckpt = torch.load(
        opt.training.volume_discriminator_path, map_location=map_location)['d']
    volume_discriminator.load_state_dict(volume_discriminator_ckpt,
                                         strict=True)

    # if opt.experiment.continue_training and
    if opt.experiment.ckpt is not None:
        if get_rank() == 0:
            print("load model:", opt.experiment.ckpt)

        ckpt_path = opt.experiment.ckpt
        # ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        ckpt = torch.load(ckpt_path, map_location=map_location)

        if opt.training.start_iter < 0:
            try:
                opt.training.start_iter = int(ckpt['iter'])
            except KeyError:
                pass

        opt.training.start_iter = max(opt.training.start_iter, 0)

        print('training start iter: ', opt.training.start_iter, flush=True)
        encoder_state_dict = encoder.state_dict()
        if 'encoder' in ckpt:
            for k, v in ckpt["encoder"].items():
                encoder_state_dict[k.replace('module.', '')] = v
        else:
            encoder_state_dict = __get_keys(ckpt, 'encoder')

        encoder.load_state_dict(encoder_state_dict, strict=True)

        print(f'load from {ckpt_path}', flush=True)
    else:
        # load pre-trained D as init of E0
        if opt.training.encoder_type == "VolumeRenderDiscriminator":
            encoder_state_dict = load_state_dict_match_size(
                volume_discriminator_ckpt, encoder.state_dict())
            encoder.load_state_dict(encoder_state_dict, strict=True)
        ckpt = None

    # todo
    if opt.training.full_pipeline:
        checkpoint_path = os.path.join('full_models',
                                       opt.experiment.expname + '.pt')
    else:
        checkpoint_path = os.path.join(
            'pretrained_renderer', f'{opt.experiment.expname}_vol_renderer.pt')

    if get_rank() == 0:
        print("load model:", checkpoint_path)
    checkpoint = torch.load(
        checkpoint_path,
        # map_location=f'cuda:{device}' if device not in ['cuda', 'cpu'] else device)
        map_location=map_location)
    generator.load_state_dict(checkpoint["g_ema"], strict=False)

    # load D
    discriminator = None
    if opt.training.adv_lambda > 0:
        discriminator = Discriminator(opt.model).to(device)
        if opt.training.Discriminator_ckpt_path is None:
            try:
                # if 'discriminator' in ckpt:
                # if 'discriminator' in ckpt:
                discriminator.load_state_dict(ckpt["discriminator"],
                                              strict=True)
                print("loading weight from ckpt Discriminator")
            except:
                discriminator.load_state_dict(checkpoint["d"], strict=True)
                print("loading weight from StyleSDF D")
        else:
            discriminator_ckpt = torch.load(
                opt.training.Discriminator_ckpt_path,
                map_location=map_location)
            # map_location=f'cuda:{device}'
            # if device not in ['cuda', 'cpu'] else device)

            D_state_dict = load_state_dict_match_size(
                discriminator_ckpt['d'], discriminator.state_dict())
            discriminator.load_state_dict(D_state_dict, True)
            print("loading weight from stylegan D")
            del discriminator_ckpt
            gc.collect()

    # * load surface sub G model
    if surface_g_ema is not None:
        # surface_extractor_dict = surface_g_ema.state_dict()
        # for k, v in checkpoint["g_ema"].items():
        #     if k in surface_extractor_dict.keys() and v.size(
        #     ) == surface_extractor_dict[k].size():
        #         surface_extractor_dict[k] = v

        # surface_g_ema.load_state_dict(surface_extractor_dict)
        surface_extractor_dict = load_state_dict_match_size(
            checkpoint["g_ema"], surface_g_ema.state_dict())

        # surface_extractor_dict = surface_g_ema.state_dict()
        # for k, v in checkpoint["g_ema"].items():
        #     if k in surface_extractor_dict.keys() and v.size(
        #     ) == surface_extractor_dict[k].size():
        #         surface_extractor_dict[k] = v

        surface_g_ema.load_state_dict(surface_extractor_dict)

    # * load original renderer.network params to now renderer.network.netGlobal
    # Load E1
    if opt.rendering.enable_local_model:
        # load original network weights to new network_Global weights
        # * first load netGlobal to g_ema and surface_g_ema
        global_renderer_dict = generator.renderer.network.netGlobal.state_dict(
        )
        prefix_to_load = 'renderer.network.'
        for k, v in checkpoint["g_ema"].items():
            key_prefix = k[:len(prefix_to_load)]
            if key_prefix != prefix_to_load:
                continue
            filtered_key = k[len(prefix_to_load):]
            if filtered_key in global_renderer_dict.keys() and v.size(
            ) == global_renderer_dict[filtered_key].size():
                global_renderer_dict[filtered_key] = v
        generator.renderer.network.netGlobal.load_state_dict(
            global_renderer_dict)

        if surface_g_ema is not None:
            surface_g_ema.renderer.network.netGlobal.load_state_dict(
                global_renderer_dict)
            print('load surface_g_ema ')

        if ckpt is not None:  # * load netLocal, to renderer in g_ema and surface_g_ema
            # if 'd' in ckpt and opt.training.discriminator_lambda > 0:
            if 'd' in ckpt and opt.training.adv_lambda > 0:
                discriminator.load_state_dict(ckpt['d'])
                print('load D from CKPT')
            if 'netLocal' in ckpt and 'netLocal' not in opt.training.ckpt_to_ignore:

                # netLocal_dict = ckpt['netLocal']  # todo
                netLocal_dict = generator.renderer.network.netLocal.state_dict()
                for k, v in ckpt['netLocal'].items():
                    if k in netLocal_dict.keys() and v.size(
                    ) == netLocal_dict[k].size():
                        netLocal_dict[k] = v

                generator.renderer.network.netLocal.load_state_dict(
                    netLocal_dict, True)
                print('load netlocal ckpt', flush=True)

                if surface_g_ema is not None:
                    surface_g_ema.renderer.network.netLocal.load_state_dict(
                        netLocal_dict, True)
                    print('load hybrid-renderer dicts for surface_g_ema')

            else:
                print("!!! didn't load netLocal ckpt, train from scratch")

    del checkpoint
    del volume_discriminator_ckpt

    # ================== Generate mean latent =======
    with torch.no_grad():
        mean_latent = generator.mean_latent(opt.inference.truncation_mean,
                                            device)
        w_space_mean_latent = mean_latent.copy()
        mean_latent[0] = mean_latent[0].repeat(
            1, opt.rendering.depth + 1).reshape(
                1, opt.rendering.depth + 1,
                mean_latent[0].shape[-1])  # 1, 256 -> 1, 9, 256
        if opt.training.full_pipeline:
            mean_latent[1] = mean_latent[1].repeat(10, 1).unsqueeze(
                0)  # 1, 512 -> 1, 10, 512
        surface_mean_latent = mean_latent[0]

    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Resize((256, 256)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
    ])

    # todo, move to eval
    # if opt.inference.evaluate:
    print(f'eval dataset: {opt.dataset.eval_dataset_path}', flush=True)
    print(f'training dataset: {opt.dataset.dataset_path}', flush=True)  # todo

    if 'ShapeNet' in opt.dataset.eval_dataset_path:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        eval_dataset = MultiResolutionDataset_ShapeNet(
            opt.dataset.eval_dataset_path, transform, opt.model.size,
            opt.model.renderer_spatial_output_dim)
    else:
        eval_dataset = ImagesDatasetEval(opt.dataset.eval_dataset_path,
                                         transform=transform)  # size 2824

    eval_loader = data.DataLoader(
        eval_dataset,
        batch_size=opt.inference.eval_batch,  # to render multi-view
        drop_last=False,
        shuffle=False,
        num_workers=1)

    test_dataset = ImagesDatasetEval(opt.dataset.test_dataset_path,
                                     transform=transform)  # size 2824
    test_loader = data.DataLoader(test_dataset,
                                  batch_size=opt.inference.eval_batch,
                                  drop_last=False,
                                  num_workers=2)

    # train_dataset = MultiResolutionDataset(opt.dataset.dataset_path, transform,
    if opt.training.lms_lambda > 0:
        lms_path = opt.dataset.lms_path
    else:
        lms_path = None
    train_dataset = MultiResolutionDatasetLMS(
        opt.dataset.dataset_path, lms_path, transform, opt.model.size,
        opt.model.renderer_spatial_output_dim)

    train_loader = data.DataLoader(train_dataset,
                                   batch_size=opt.training.batch,
                                   sampler=data_sampler(
                                       train_dataset,
                                       shuffle=True,
                                       distributed=opt.training.distributed),
                                   drop_last=True,
                                   num_workers=opt.training.workers,
                                   pin_memory=True)
    # num_workers=8)

    if get_rank() == 0 and opt.training.wandb:
        import wandb

        # and not opt.inference.evaluate and wandb is not None and opt.training.wandb:
        if opt.training.test_optimisation:
            wandb.init(project="3dinversion", resume=False)
        else:
            os.environ[
                "WANDB_START_METHOD"] = "thread"  # * only in multiprocessing
            wandb.init(project="e3dge", resume=False)
        wandb.run.name = '/'.join(opt.training.checkpoints_dir.split('/')[-2:])
        wandb.config.update(opt.rendering, allow_val_change=True)
        wandb.config.update(opt.model, allow_val_change=True)
        wandb.config.update(opt.training, allow_val_change=True)
    else:
        wandb = None
    # writer = SummaryWriter()

    gc.collect()
    torch.cuda.empty_cache()

    print('start training', flush=True)
    opt.training.sample_near_surface = opt.rendering.sample_near_surface
    print(os.environ['CUDA_VISIBLE_DEVICES'])

    # save opt

    with open(exp_root / 'opt.json', 'w') as f:
        json.dump(dict(opt), f, indent=4, sort_keys=True)

    all_loaders = {
        'train': train_loader,
        'val': eval_loader,
        'test': test_loader
    }

    # if opt.training.adv_lambda > 0:
    return all_loaders, generator, discriminator, surface_g_ema, encoder, volume_discriminator, loss_class, mean_latent, surface_mean_latent, ckpt
