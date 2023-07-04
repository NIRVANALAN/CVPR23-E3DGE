# for 3dae training or test-time-optimisation
import mmcv
import os

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

    # https://github.com/pytorch/pytorch/blob/master/torch/distributed/launch.py
    # device = {
    #     'cuda:%d' % 0: 'cuda:%d' % opt.training.local_rank
    # }  # configure map_location properly
    device = 'cuda:{}'.format(opt.training.local_rank)
    torch.cuda.set_device(opt.training.local_rank)

    torch.manual_seed(opt.training.seed)

    if opt.training.adv_lambda > 0:
        loader, generator, discriminator, surface_g_ema, encoder, view_encoder, loss_class, mean_latent, surface_mean_latent, loaded_ckpt = setup_training_configs(
            opt, device)
    else:
        loader, generator, discriminator, surface_g_ema, encoder, view_encoder, loss_class, mean_latent, surface_mean_latent, loaded_ckpt = setup_training_configs(
            opt, device)
        # discriminator = None

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
        mode='train' if not opt.training.test_optimisation else 'val',
        device=device,
        discriminator=discriminator)

    trainer = mmcv.build_from_cfg(runner_cfg, RUNNER)
    trainer.run()
    if not opt.training.test_optimisation:
        trainer.validation(mode='test')
    # except Exception:
    #     print(traceback.format_exc())
