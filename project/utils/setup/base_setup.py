import os
import sys
from project.utils.options import BaseOptions

from pathlib import Path

# for autoencoder
from project.models.op import grid_sample_gradfix

sys.path.append('project/vendor/pifu')

# pifu modules
from lib.options import BaseOptionsPiFU
# from lib.mesh_util import *
# from lib.sample_util import *
# from lib.train_util import *
# from lib.data import *
# from lib.model import HGPIFuNetGAN


def setup_opts(args=None):  # todo
    # ================== load local prior training options =========
    local_prior_parser = BaseOptionsPiFU().get_parser()

    # ================== load global prior training options =========
    base_parser = BaseOptions(local_prior_parser)

    # base_parser.merge_parser(merge_parser, merge_parser_name)

    opt = base_parser.parse(filter_key='pifu', args=args)
    opt.model.is_test = False
    opt.model.freeze_renderer = False
    opt.rendering.camera = opt.camera
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
    # opt.training.start_iter = 0

    opt.training.eval_dataset_path = opt.dataset.eval_dataset_path
    opt.training.save_img = opt.inference.save_img

    # for rendering
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

    # for renderer training
    opt.training.with_sdf = not opt.rendering.no_sdf
    if opt.training.with_sdf and opt.training.min_surf_lambda > 0:
        opt.rendering.return_sdf = True
        opt.training.iter = 600

    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    opt.training.distributed = n_gpu > 1

    opt.rendering.pifu = opt.pifu

    results_dir_basename = os.path.join(opt.training.checkpoints_dir,
                                        opt.experiment.expname)
    root_dir = Path(results_dir_basename)
    opt.training.results_dst_dir = str(root_dir / 'train')
    opt.inference.results_dst_dir = str(root_dir / 'eval')

    # ==================create results directory ==================
    os.makedirs(opt.training.results_dst_dir, exist_ok=True)
    os.makedirs(os.path.join(opt.training.results_dst_dir, 'images'),
                exist_ok=True)
    # === setting up flags
    if opt.training.enable_custom_grid_sample:
        grid_sample_gradfix.enabled = True
        print('enable nv grid-sample op', flush=True)

    return opt
