# import argparse
import configargparse as argparse


# class BaseOptions():
class BaseOptionsPiFU():

    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        # Datasets related
        g_data = parser.add_argument_group('pifuData')
        g_data.add_argument('--dataroot',
                            type=str,
                            default='./data',
                            help='path to images (data folder)')

        g_data.add_argument('--loadSize',
                            type=int,
                            default=256,
                            help='load size of input image')

        # Experiment related
        g_exp = parser.add_argument_group('pifuExperiment')
        g_exp.add_argument(
            '--name',
            type=str,
            default='example',
            help=
            'name of the experiment. It decides where to store samples and models'
        )
        g_exp.add_argument('--debug',
                           action='store_true',
                           help='debug mode or not')
        g_exp.add_argument('--return_eikonal',
                           action='store_true',
                           help='return normal for suprevision')
        g_exp.add_argument('--enforce_minmax',
                           action='store_true',
                           help='debug mode or not')
        # g_exp.add_argument('--add_fg_mask', action='store_true', help='debug mode or not')

        g_exp.add_argument('--num_views',
                           type=int,
                           default=1,
                           help='How many views to use for multiview network.')
        g_exp.add_argument('--random_multiview',
                           action='store_true',
                           help='Select random multiview combination.')

        # debugging
        g_exp.add_argument('--fix_beta',
                           action='store_true',
                           help='fix learnable bce beta')

        # Training related
        g_train = parser.add_argument_group('pifuTraining')
        g_train.add_argument('--gpu_id',
                             type=int,
                             default=0,
                             help='gpu id for cuda')
        g_train.add_argument(
            '--gpu_ids',
            type=str,
            default='0',
            help='gpu ids: e.g. 0  0,1,2, 0,2, -1 for CPU mode')

        g_train.add_argument('--num_threads',
                             default=1,
                             type=int,
                             help='# sthreads for loading data')
        g_train.add_argument(
            '--serial_batches',
            action='store_true',
            help=
            'if true, takes images in order to make batches, otherwise takes them randomly'
        )
        g_train.add_argument('--pin_memory',
                             action='store_true',
                             help='pin_memory')

        # g_train.add_argument('--batch_size', type=int, default=2, help='input batch size')
        g_train.add_argument('--learning_rate',
                             type=float,
                            #  default=1e-4,
                            #  default=1e-15, # debugging fix netLocal here
                            #  default=5e-5, # debugging fix netLocal here
                             default=5e-5, # debugging fix netLocal here
                             help='adam learning rate')  # todo, decay
        g_train.add_argument('--num_epoch',
                             type=int,
                             default=100,
                             help='num epoch to train')
        g_train.add_argument('--num_iter',
                             type=int,
                             default=1e6,
                             help='num iterations to train')

        g_train.add_argument('--freq_plot',
                             type=int,
                             default=1000,
                             help='freqency of the error plot')
        g_train.add_argument('--freq_save',
                             type=int,
                             default=5000,
                             help='freqency of the save_checkpoints')
        g_train.add_argument('--freq_save_ply',
                             type=int,
                             default=100,
                             help='freqency of the save ply')
        g_train.add_argument('--freq_eval',
                             type=int,
                             default=10,
                             help='freqency of the error plot')

        g_train.add_argument('--no_gen_mesh', action='store_true')
        g_train.add_argument('--no_num_eval', action='store_true')

        g_train.add_argument('--resume_epoch',
                             type=int,
                             default=-1,
                             help='epoch resuming the training')
        g_train.add_argument('--continue_train',
                             action='store_true',
                             help='continue training: load the latest model')

        # Testing related
        g_test = parser.add_argument_group('pifuTesting')
        g_test.add_argument('--resolution',
                            type=int,
                            default=256,
                            help='# of grid in mesh reconstruction')
        g_test.add_argument('--test_folder_path',
                            type=str,
                            default=None,
                            help='the folder of test image')
        g_train.add_argument(
            '--uniform_pts_loss',
            type=str,
            default='mse',
            help='how to supervise the in/out poitns, pifu uses mse')

        # Sampling related
        g_sample = parser.add_argument_group('pifuSampling')
        g_sample.add_argument(
            '--sigma',
            type=float,
            default=5.0,
            help='perturbation standard deviation for positions')

        g_sample.add_argument('--num_sample_inout',
                              type=int,
                              default=5000,
                              help='# of sampling points')
        g_sample.add_argument('--num_sample_color',
                              type=int,
                              default=0,
                              help='# of sampling points')

        # g_sample.add_argument('--z_size', type=float, default=200.0, help='z normalization factor')
        g_sample.add_argument('--z_size',
                              type=float,
                              default=1.12,
                              help='z normalization factor')

        # Model related
        g_model = parser.add_argument_group('pifuModel')
        # General
        g_model.add_argument(
            '--norm',
            type=str,
            default='group',
            help=
            'instance normalization or batch normalization or group normalization'
        )
        g_model.add_argument(
            '--norm_color',
            type=str,
            default='instance',
            help=
            'instance normalization or batch normalization or group normalization'
        )

        # hg filter specify
        g_model.add_argument('--num_stack',
                             type=int,
                             default=4,
                             help='# of hourglass')
        g_model.add_argument('--num_hourglass',
                             type=int,
                             default=2,
                             help='# of stacked layer of hourglass')
        g_model.add_argument('--skip_hourglass',
                             action='store_true',
                             help='skip connection in hourglass')
        g_model.add_argument('--hg_input_channel',
                             type=int,
                             default=3,
                             help='num of in channel')
        g_model.add_argument('--hg_down',
                             type=str,
                             default='ave_pool',
                             help='ave pool || conv64 || conv128')
        g_model.add_argument('--hourglass_dim',
                             type=int,
                             default='256',
                             help='256 | 512')

        # Classification General
        g_model.add_argument('--mlp_dim',
                             nargs='+',
                             default=[257, 1024, 512, 256, 128, 1],
                             type=int,
                             help='# of dimensions of mlp')
        g_model.add_argument('--mlp_dim_color',
                             nargs='+',
                             default=[513, 1024, 512, 256, 128, 3],
                             type=int,
                             help='# of dimensions of color mlp')

        g_model.add_argument(
            '--use_tanh',
            action='store_true',
            help='using tanh after last conv of image_filter network')

        # for train
        g_train = parser.add_argument_group('pifuTrain')
        g_train.add_argument('--random_flip',
                             action='store_true',
                             help='if random flip')
        g_train.add_argument('--random_trans',
                             action='store_true',
                             help='if random flip')
        g_train.add_argument('--random_scale',
                             action='store_true',
                             help='if random flip')
        g_train.add_argument('--no_residual',
                             action='store_true',
                             help='no skip connection in mlp')
        g_train.add_argument('--schedule',
                             type=int,
                             nargs='+',
                             default=[60, 80],
                             help='Decrease learning rate at these epochs.')
        g_train.add_argument('--gamma',
                             type=float,
                             default=0.1,
                             help='LR is multiplied by gamma on schedule.')
        g_train.add_argument('--color_loss_type',
                             type=str,
                             default='l1',
                             help='mse | l1')

        g_train.add_argument('--lambda_g1', type=float, default=1, help='')
        g_train.add_argument('--lambda_g2', type=float, default=1, help='')
        g_train.add_argument('--lambda_l', type=float, default=0.2, help='')
        g_train.add_argument('--lambda_e', type=float, default=0.1, help='')

        # for eval
        g_eval = parser.add_argument_group('pifuEval')
        g_eval.add_argument('--val_test_error',
                            action='store_true',
                            help='validate errors of test data')
        g_eval.add_argument('--val_train_error',
                            action='store_true',
                            help='validate errors of train data')
        g_eval.add_argument('--gen_test_mesh',
                            action='store_true',
                            help='generate test mesh')
        g_eval.add_argument('--gen_train_mesh',
                            action='store_true',
                            help='generate train mesh')
        g_eval.add_argument('--all_mesh',
                            action='store_true',
                            help='generate meshs from all hourglass output')
        g_eval.add_argument('--num_gen_mesh_test',
                            type=int,
                            default=1,
                            help='how many meshes to generate during testing')

        # path
        g_path = parser.add_argument_group('pifuPath')
        g_path.add_argument('--checkpoints_path',
                            type=str,
                            default='./checkpoints',
                            help='path to save checkpoints')
        g_path.add_argument('--load_netG_checkpoint_path',
                            type=str,
                            default=None,
                            help='path to save checkpoints')
        g_path.add_argument('--load_netC_checkpoint_path',
                            type=str,
                            default=None,
                            help='path to save checkpoints')
        g_path.add_argument('--results_path',
                            type=str,
                            default='./results',
                            help='path to save results ply')
        g_path.add_argument('--load_checkpoint_path',
                            type=str,
                            help='path to save results ply')
        g_path.add_argument('--single',
                            type=str,
                            default='',
                            help='single data for training')
        # for single image reconstruction
        g_path.add_argument('--mask_path',
                            type=str,
                            help='path for input mask')
        g_path.add_argument('--img_path',
                            type=str,
                            help='path for input image')

        # aug
        g_aug = parser.add_argument_group(
            'pifuaug')  # * just used in help messages
        g_aug.add_argument('--aug_alstd',
                           type=float,
                           default=0.0,
                           help='augmentation pca lighting alpha std')
        g_aug.add_argument('--aug_bri',
                           type=float,
                           default=0.0,
                           help='augmentation brightness')
        g_aug.add_argument('--aug_con',
                           type=float,
                           default=0.0,
                           help='augmentation contrast')
        g_aug.add_argument('--aug_sat',
                           type=float,
                           default=0.0,
                           help='augmentation saturation')
        g_aug.add_argument('--aug_hue',
                           type=float,
                           default=0.0,
                           help='augmentation hue')
        g_aug.add_argument('--aug_blur',
                           type=float,
                           default=0.0,
                           help='augmentation blur')

        # special tasks
        self.initialized = True
        return parser

    def get_parser(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        return parser

    # def parse_args(self):

    # self.parser = parser

    # return self.parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def parse(self):
        parser = self.get_parser()
        self.parser = parser
        opt = self.parser.parse_args()
        return opt
