# for 3dae training or test-time-optimisation
import mmcv
import os
from pdb import set_trace as st
import torch
from project.utils.dist_utils import get_rank
from project.utils.setup.base_setup import setup_opts  # todo
from project.utils.setup.train_setup import setup_training_configs

from project.trainers.base_runner import RUNNER

if __name__ == "__main__":

    if get_rank() == 0:
        print('using gpu: ', os.environ['CUDA_VISIBLE_DEVICES'])
    os.environ[
        "TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"  # set to DETAIL for runtime logging.
    os.environ["TORCH_CPP_LOG_LEVEL"] = "INFO"

    torch.autograd.set_detect_anomaly(True)
    opt = setup_opts()  # todo
    device = 'cuda'

    torch.manual_seed(opt.training.seed)

    loader, generator, discriminator, surface_g_ema, encoder, view_encoder, loss_class, mean_latent, surface_mean_latent, loaded_ckpt = setup_training_configs(
        opt, device)

    runner_cfg = dict(
        type=opt.training.runner,
        encoder=encoder,
        volume_discriminator=view_encoder,
        generator=generator,
        mean_latent=mean_latent,
        loaders=loader,
        loss_class=loss_class,
        surface_g_ema=surface_g_ema,
        ckpt=loaded_ckpt,
        opt=opt,
        mode='val',
        # mode=opt.inference.mode,
        device=device,
        discriminator=discriminator)

    trainer = mmcv.build_from_cfg(runner_cfg, RUNNER)
    # if opt.inference.no_surface_renderings:
    if opt.inference.deca_eval:
        trainer.evaluate3D(mode=opt.inference.mode)

    elif opt.inference.trajectory_eval:
        trainer.evaluateTrajectory()

    elif opt.inference.nvs_video:
        trainer.render_HDTF()

    else:
        trainer.validation()
    # else:
