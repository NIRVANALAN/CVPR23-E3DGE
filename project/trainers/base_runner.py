# Copyright (c) OpenMMLab. All rights reserved.
# also modified from https://github.com/BasicSR
# lint version of BaseRunner
from pathlib import Path
import os
import os.path as osp
from abc import abstractmethod

import torch
import shutil
import mmcv
# from .checkpoint import load_checkpoint
from mmcv.runner import get_time_str, get_dist_info

RUNNER = mmcv.Registry('runner')


# class BaseRunner(metaclass=ABCMeta):
@RUNNER.register_module()
class BaseRunner():
    """The base class of Runner, a training helper for PyTorch.
    Removed all dependencies to hooks

    All subclasses should implement the following APIs:

    - ``run()``
    - ``train()``
    - ``val()``
    - ``save_checkpoint()``

    Args:
        model (:obj:`torch.nn.Module`): The model to be run.
        batch_processor (callable): A callable method that process a data
            batch. The interface of this method should be
            `batch_processor(model, data, train_mode) -> dict`
        optimizer (dict or :obj:`torch.optim.Optimizer`): It can be either an
            optimizer (in most cases) or a dict of optimizers (in models that
            requires more than one optimizer, e.g., GAN).
        work_dir (str, optional): The working directory to save checkpoints
            and logs. Defaults to None.
        logger (:obj:`logging.Logger`): Logger used during training.
             Defaults to None. (The default value is just for backward
             compatibility)
        meta (dict | None): A dict records some import information such as
            environment info and seed, which will be logged in logger hook.
            Defaults to None.
        max_epochs (int, optional): Total training epochs.
        max_iters (int, optional): Total training iterations.
    """

    def __init__(self, encoder, opt, work_dir=None, max_iters=None, max_epochs=None):

        self.opt = opt
        self.train_opt = opt.training
        self.experiment_opt = opt.experiment

        self.encoder = encoder
        # self.model = model
        self.optimizers = {}
        self.network = {}
        self.network = {}

        if mmcv.is_str(work_dir):
            self.work_dir = osp.abspath(work_dir)
            mmcv.mkdir_or_exist(self.work_dir)
        elif work_dir is None:
            self.work_dir = None
        else:
            raise TypeError('"work_dir" must be a str or None')

        self._rank, self._world_size = get_dist_info()
        self.timestamp = get_time_str()
        self.mode = None

        self._epoch = 0
        self._iter = 0
        self._inner_iter = 0

        if max_epochs is not None and max_iters is not None:
            raise ValueError(
                'Only one of `max_epochs` or `max_iters` can be set.')

        self._max_epochs = max_epochs
        self._max_iters = max_iters

    @property
    def model_name(self):
        """str: Name of the model, usually the module class name."""
        return self._model_name

    @property
    def rank(self):
        """int: Rank of current process. (distributed training)"""
        return self._rank

    @property
    def world_size(self):
        """int: Number of processes participating in the job.
        (distributed training)"""
        return self._world_size

    @property
    def hooks(self):
        """list[:obj:`Hook`]: A list of registered hooks."""
        return self._hooks

    @property
    def epoch(self):
        """int: Current epoch."""
        return self._epoch

    @property
    def iter(self):
        """int: Current iteration."""
        return self._iter

    @property
    def inner_iter(self):
        """int: Iteration in an epoch."""
        return self._inner_iter

    @property
    def max_epochs(self):
        """int: Maximum training epochs."""
        return self._max_epochs

    @property
    def max_iters(self):
        """int: Maximum training iterations."""
        return self._max_iters

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def val(self):
        pass

    def run(self) -> None:
        """Launch training."""
        # self.runner.call_hook('before_train')
        # # In iteration-based training loop, we treat the whole training process
        # # as a big epoch and execute the corresponding hook.
        # self.runner.call_hook('before_train_epoch')
        # while self._iter < self._max_iters:
        #     self.runner.model.train()
        #
        #     data_batch = next(self.dataloader_iterator)
        #     self.run_iter(data_batch)
        #
        #     self._decide_current_val_interval()
        #     if (self.runner.val_loop is not None
        #             and self._iter >= self.val_begin
        #             and self._iter % self.val_interval == 0):
        #         self.runner.val_loop.run()
        #
        # self.runner.call_hook('after_train_epoch')
        # self.runner.call_hook('after_train')
        # return self.runner.model

    # @abstractmethod
    # def save(self,
    #                     out_dir,
    #                     filename_tmpl,
    #                     save_optimizer=True,
    #                     meta=None,
    #                     create_symlink=True):
    #     pass

    @abstractmethod
    def train_step():
        """
        ret:
            ret_dict = dict(noise=noise,
                        random_3d_sample_batch=random_3d_sample_batch,
                        render_out=render_out)
        """

    @abstractmethod
    def _build_model(self):
        pass

    def save_training_state(self):
        pass

    def resume_training(self, resume_state):
        pass

    def zero_grad(self):
        # super().zero_grad()
        for _, v in self.network.items():
            v.zero_grad()

    # todo, move to parent
    def train_mode(self):
        # super().train_mode()
        for _, v in self.network.items():
            v.train()

    def eval_mode(self):
        # super().eval_mode()
        for _, v in self.network.items():
            v.eval()

    def _dist_prepare(self):
        opt = self.opt.training
        # super()._dist_prepare()
        self.network_module = dict()

        # todo, move to base_runner if lms works
        for k, v in self.network.items():
            if opt.distributed:
                v = torch.nn.parallel.DistributedDataParallel(
                    v,
                    device_ids=[opt.local_rank],
                    output_device=opt.local_rank,
                    broadcast_buffers=False,
                    find_unused_parameters=False)
                # * for saving
                self.network_module[k] = v.module
            else:
                self.network_module[k] = v

    def _get_trainable_parmas(self):
        opt = self.opt.training
        parent_params = []

        for k, v in self.network.items():
            optim_params = []

            for param_k, param_v in v.parameters():
                if param_v.requires_grad:
                    optim_params.append(param_v)

            if f'{k}_lr' in opt:
                lr = opt["f'{k}_lr'"]
            else:
                # lr = opt.ada_lr
                lr = opt.lr
            print('using lr: {}, network: {}'.format(lr, k))

            params_group = {
                'name': k,
                'params': optim_params,
                'lr': lr  # todo
            }
            parent_params.extend([params_group])

        return parent_params

    @torch.no_grad()
    def save_network(self, filename_tmpl=None):
        save_dict = {
            'iter': self._iter,
        }

        for model_name, model_params in self.network.items():
            if model_name not in save_dict and model_name not in self.opt.training.ckpt_to_ignore:
                state_dict = model_params.state_dict()

                for key, param in state_dict.items():
                    if key.startswith(
                            'module.'):  # remove unnecessary 'module.'
                        key = key[7:]
                    state_dict[key] = param.cpu()
                save_dict[model_name] = state_dict

        # save parameters
        for optim_k, optim_v in self.optimizers.items():
            save_dict.update({f'{optim_k}_optimizer': optim_v.state_dict()})

        ckpt_save_path = Path(
            os.path.join(self.opt.training.checkpoints_dir,
                         f"models_{filename_tmpl}.pt"))

        # move to olt.pt for back up
        if ckpt_save_path.exists():
            backup_ckpt_save_path = Path(
                os.path.join(self.opt.training.checkpoints_dir,
                             f"models_{filename_tmpl}_old.pt"))
            shutil.move(ckpt_save_path, backup_ckpt_save_path)

        torch.save(save_dict, ckpt_save_path)
        print('saved: ', save_dict.keys())
