import configargparse
from munch import Munch


class BaseOptions():

    def __init__(self, parent_parser=None):
        if parent_parser is None:
            self.parser = configargparse.ArgumentParser()
        else:
            self.parser = parent_parser

        self.parser.add('--config_file',
                        is_config_file=True,
                        help='config file path')

        self.initialized = False
        self.opt = None

    def initialize(self):
        configs = self.parser.add_argument_group('configs')
        configs.add_argument("--config_paths",
                             nargs="+",
                             type=str,
                             help="configuration files list")

        # Dataset options
        dataset = self.parser.add_argument_group('dataset')
        dataset.add_argument(
            "--dataset_path",
            type=str,
            default='/mnt/lustre/share/shuaiyang/ffhq/realign256x256/',
            help="path to the lmdb dataset")
        dataset.add_argument(
            "--lms_path",
            type=str,
            default='/mnt/lustre/share/yslan/ffhq/orig_data/1024/lms_gt_98.pkl',
            help="path to the predicted lms dataset")
        dataset.add_argument("--test_dataset_path",
                             type=str,
                             default='',
                             help="path to the test dataset")

        dataset.add_argument(
            "--eval_dataset_path",
            type=str,
            default=
            '/mnt/lustre/share/yslan/CelebAMask-HQ/CelebA-HQ-img-subset/',
            help="path to the lmdb dataset")

        # Experiment Options
        experiment = self.parser.add_argument_group('experiment')
        experiment.add_argument('--config',
                                is_config_file=True,
                                help='config file path')
        experiment.add_argument("--port",
                                type=int,
                                default=12345,
                                help="total number of training iterations")
        experiment.add_argument("--expname",
                                type=str,
                                default='ffhq1024x1024',
                                help='experiment name')
        experiment.add_argument(
            "--ckpt_2DAlignment",
            type=str,
            default=None,
            help="path to the checkpoints to resume training, for 2D ALignment"
        )
        experiment.add_argument(
            "--ckpt",
            type=str,
            default=None,
            help="path to the checkpoints to resume training")
        experiment.add_argument("--continue_training",
                                action="store_true",
                                help="continue training the model")

        # Training loop options
        training = self.parser.add_argument_group('training')

        # synthetic only
        training.add_argument("--synthetic_sampling_strategy",
                              type=str,
                              default='half',
                              help="how to use synthetic sampling training?")

        training.add_argument("--checkpoints_dir",
                              type=str,
                              default='./checkpoint',
                              help='checkpoints directory name')
        training.add_argument("--iter",
                              type=int,
                              default=500000,
                              help="total number of training iterations")
        training.add_argument(
            "--batch",
            type=int,
            default=4,
            help=
            "batch sizes for each GPU. A single RTX2080 can fit batch=4, chunck=1 into memory."
        )
        training.add_argument(
            "--chunk",
            type=int,
            default=4,
            help=
            'number of samples within a batch to processed in parallel, decrease if running out of memory'
        )
        training.add_argument(
            "--val_n_sample",
            type=int,
            default=8,
            help="number of test samples generated during training")
        training.add_argument(
            "--d_reg_every",
            type=int,
            default=16,
            help=
            "interval for applying r1 regularization to the StyleGAN generator"
        )
        training.add_argument(
            "--g_reg_every",
            type=int,
            default=4,
            help=
            "interval for applying path length regularization to the StyleGAN generator"
        )
        training.add_argument("--local_rank",
                              type=int,
                              default=0,
                              help="local rank for distributed training")
        training.add_argument("--mixing",
                              type=float,
                              default=0.9,
                              help="probability of latent code mixing")
        training.add_argument(
            "--D_aligned_res",
            action='store_true',
            help='add aligned_res to D loss',
        )
        training.add_argument(
            "--aligner_norm_type",
            default='batch',
            choices=['batch', 'instance', 'none'],
            # action="store_true",
            type=str,
            help="BN/IN/None")
        training.add_argument("--aligner_demodulate",
                              action="store_true",
                              help="try to fix droplet issue")
        training.add_argument(
            "--ada_lr",
            type=float,
            default=1e-4,  # follow hfgi
            help="learning rate of ADA()")
        training.add_argument("--lr",
                              type=float,
                              default=5e-5,
                              help="learning rate")
        training.add_argument("--r1",
                              type=float,
                              default=10,
                              help="weight of the r1 regularization")
        training.add_argument("--view_lambda",
                              type=float,
                              default=15,
                              help="weight of the viewpoint regularization")
        training.add_argument(
            "--surf_sdf_lambda",
            type=float,
            default=1,  # todo, graduall increase to 15
            help="weight of the surface sdf reconstrution regularization")
        training.add_argument(
            "--surf_normal_lambda",
            type=float,
            default=1,
            help="weight of the surface normal reconstrution regularization")
        training.add_argument(
            "--uniform_pts_sdf_lambda",
            type=float,
            default=0.,
            help=
            "weight of the uniform sampled poitns sdf reconstrution regularization"
        )
        training.add_argument("--eikonal_lambda",
                              type=float,
                              default=0.,
                              help="weight of the eikonal regularization")
        training.add_argument("--chamfer_distance_lambda",
                              type=float,
                              default=0.,
                              help="weight of the eikonal regularization")
        training.add_argument(
            "--min_surf_lambda",
            type=float,
            default=0.05,
            help="weight of the minimal surface regularization")
        training.add_argument(
            "--min_surf_beta",
            type=float,
            default=100.0,
            help="weight of the minimal surface regularization")
        training.add_argument("--path_regularize",
                              type=float,
                              default=2,
                              help="weight of the path length regularization")
        training.add_argument(
            "--path_batch_shrink",
            type=int,
            default=2,
            help=
            "batch size reducing factor for the path length regularization (reduce memory consumption)"
        )
        training.add_argument("--wandb",
                              action="store_true",
                              help="use weights and biases logging")
        training.add_argument(
            "--no_sphere_init",
            action="store_true",
            help="do not initialize the volume renderer with a sphere SDF")

        # extra training arguments for auto-encoder
        training.add_argument("--fix_G",
                              action="store_true",
                              help="fix gradient of generator")

        training.add_argument("--fix_netGlobal",
                              action="store_true",
                              help="fix gradient of renderer")
        training.add_argument("--fix_renderer",
                              action="store_true",
                              help="fix gradient of all renderers")

        training.add_argument("--fix_decoder",
                              action="store_true",
                              help="fix gradient of decoder")
        # grad flags
        training.add_argument("--G_renderer_grad_false",
                              action="store_true",
                              help="fix gradient")
        training.add_argument("--G_decoder_grad_false",
                              action="store_true",
                              help="fix gradient")
        training.add_argument("--E_l_grad_false",
                              action="store_true",
                              help="fix gradient of E1")
        training.add_argument("--E_g_grad_false",
                              action="store_true",
                              help="fix gradient of the parameters corresponding to the renderer part (pigan styles)")
        training.add_argument("--E_d_grad_false",
                              action="store_true",
                              help="fix gradient of Discriminator")
        training.add_argument("--fix_E_fintune_last",
                              action="store_true",
                              help="fix w")
        training.add_argument("--E_backbone_false",
                              action="store_true",
                              help="fix all encoder grads")
        training.add_argument("--fix_stylegan_encoder",
                              action="store_true",
                              help="fix stylegan encoder")
        training.add_argument("--fix_renderer_encoder",
                              action="store_true",
                              help="fix renderer encoder")

        training.add_argument("--fix_ada", action="store_true", help="fix ada")

        # more training flags for 3d consistent training
        training.add_argument("--local_prior",
                              action="store_true",
                              help="train local-prior network only")
        training.add_argument(
            "--dry_run",
            action="store_true",
            help="train local-prior and global-prior networks only")
        training.add_argument(
            "--local_gloabl_prior",
            action="store_true",
            help="train local-prior and global-prior networks only")
        training.add_argument('--load_local_netG_checkpoint_path',
                              default=None,
                              type=str,
                              help='local-prior model ckpt')
        # local branch flags
        training.add_argument("--test_optimisation",
                              action="store_true",
                              help="test_optimisation")
        # cycle consistent flags
        training.add_argument("--swap_code",
                              action="store_true",
                              help="for code geometry consistency")
        training.add_argument("--swap_res",
                              action="store_true",
                              help="for training ada")
        # 2d matching
        training.add_argument("--binary_mask", action='store_true', help="")
        training.add_argument(
            "--visibility_threshold",
            type=float,
            default=0.5,  # follow hfgi
            help="if the queried visibility < this value, mask out.")

        # Inference Options
        inference = self.parser.add_argument_group('inference')
        # inference.add_argument("--eval_mode",
        #                        type=str,
        #                        default='train',
        #                        help='test/val/train')
        inference.add_argument("--editing_inference",
                               action="store_true",
                               help="for editing")
        inference.add_argument(
            "--editing_boundary_dir",
            type=str,
            default='assets/editing_boundaries/stylesdf',
            help='dir of the editing directions.')
        inference.add_argument("--editing_boundary_scale_upperbound",
                               nargs="+",
                               type=float,
                               default=[2, 1.5, 1.5, 2, 0],
                               help="scale to apply the directions")
        inference.add_argument("--editing_boundary_scale_lowerbound",
                               nargs="+",
                               type=float,
                               default=[-1.5, -1.4, -1.5, -2, 0],
                               help="scale to apply the directions")
        inference.add_argument("--editing_boundary_scale",
                               nargs="+",
                               type=float,
                               default=[0, 1, 0, 0, 0],
                               help="scale to apply the directions")

        inference.add_argument("--smile_ids",
                               nargs="+",
                               type=str,
                               default=[322, 438, 443, 468, 485, 532],
                               help="identities in CelebA-HQ for smiling editing visualization")

        inference.add_argument("--beard_ids",
                               nargs="+",
                               type=str,
                               default=[320,372,540],
                               help="identities in CelebA-HQ for beard editing visualization")

        inference.add_argument("--bangs_ids",
                               nargs="+",
                               type=str,
                               default=[307,481,543],
                               help="identities in CelebA-HQ for bangs editing visualization")

        inference.add_argument("--age_ids",
                               nargs="+",
                               type=str,
                               default=[494,519,524,581,587],
                               help="identities in CelebA-HQ for age editing visualization")


        inference.add_argument("--results_dir",
                               type=str,
                               default='./evaluations',
                               help='results/evaluations directory name')

        inference.add_argument(
            "--editing_relative_scale",
            type=float,
            default=-1,
            help="for easy editing, maximum is 1, reletive to min - max bound")

        inference.add_argument("--stylemix",
                               action="store_true",
                               help="for analysis")
        inference.add_argument("--show",
                               action="store_true",
                               help="save the imgs locally")
        inference.add_argument("--shading",
                               action="store_true",
                               help="shade marching cube results")
        inference.add_argument("--output_id_loss",
                               action="store_true",
                               help='to evaluate trajectory consistency')
        inference.add_argument("--mode",
                               type=str,
                               default='lpips',
                               help='evaluation metrics')
        inference.add_argument("--result_model_dir",
                               type=str,
                               default='final_model',
                               help='results/evaluations/? directory name')
        # inference.add_argument(
        #     "--inverse_image_path",
        #     type=str,
        #     default=
        #     'evaluations/ffhq1024x1024/vis-geometry/fixed_angles/images/0000000_gt.png',
        #     help='for inversion image path')
        inference.add_argument("--fixed_latents_path",
                               type=str,
                               default=None,
                               help="inverse specific latents")
        inference.add_argument("--inverse_space",
                               type=str,
                               default='w_plus',
                               help="path to image files to be projected")
        inference.add_argument(
            "--truncation_ratio",
            type=float,
            default=0.55,
            help=
            "truncation ratio, controls the diversity vs. quality tradeoff. Higher truncation ratio would generate more diverse results"
        )
        inference.add_argument(
            "--video_frames",
            type=int,
            default=250,
            help="number of vectors to calculate mean for the truncation")
        inference.add_argument(
            "--truncation_mean",
            type=int,
            default=10000,
            help="number of vectors to calculate mean for the truncation")
        inference.add_argument("--identities",
                               type=int,
                               default=1,
                               help="number of identities to be generated")
        inference.add_argument(
            "--num_views_per_id",
            type=int,
            default=1,
            help="number of viewpoints generated per identity")
        inference.add_argument("--no_eval",
                               action="store_true",
                               help="when true, no validation")
        inference.add_argument("--render_multiview",
                               action="store_true",
                               help="when true, call render_multiview()")
        inference.add_argument(
            "--surface_rendering_marching_cube",
            action="store_true",
            help=
            "when true, only RGB outputs will be generated. otherwise, both RGB and depth videos/renderings will be generated. this cuts the processing time per video"
        )
        inference.add_argument(
            "--render_video",
            action="store_true",
            help="when true, render tex and geo reconstruction video")
        inference.add_argument(
            "--no_surface_renderings",
            action="store_true",
            help=
            "when true, only RGB outputs will be generated. otherwise, both RGB and depth videos/renderings will be generated. this cuts the processing time per video"
        )
        inference.add_argument(
            "--fixed_camera_angles",
            action="store_true",
            help=
            "when true, the generator will render indentities from a fixed set of camera angles."
        )
        inference.add_argument(
            "--eval_batch",
            type=int,
            default=16,
            help=
            "batch sizes for each GPU for inference. BS=16 takes 29GB GPU Mem")
        inference.add_argument(
            "--few_samples",
            action="store_true",
            help=
            "when true, the generator will render indentities from a fixed set of camera angles."
        )
        inference.add_argument(
            "--evaluate",
            action="store_true",
            help=
            "when true, the generator will render indentities from a fixed set of camera angles."
        )
        inference.add_argument(
            "--evaluate_chamfer",
            action="store_true",
            help="when true, the generator will evaluate chamfer.")
        # trajectory evaluation
        inference.add_argument(
            "--trajectory_eval",
            action="store_true",
            help="when true, output trajectory for evaluation.")

        inference.add_argument(
            "--nvs_video",
            action="store_true",
            help="when true, output nvs trajectory for visualization.")

        inference.add_argument("--video_output_csv",
                               type=str,
                               default='',
                               help='list of ids to generate video')

        inference.add_argument(
            "--trajectory_gt_root",
            type=str,
            default=
            '/mnt/lustre/yslan/Dataset/CVPR23/TRAJECTORIES_FOR_EVALUATION/',
            help='GT for 500 trajectories')

        inference.add_argument(
            "--deca_eval",
            action="store_true",
            help="when true, the generator will evaluate chamfer.")
        inference.add_argument("--gt_mesh_folder",
                               type=str,
                               default='',
                               help='gt_mesh_folder')
        inference.add_argument("--pred_mesh_folder",
                               type=str,
                               default='',
                               help='pred_mesh_folder')
        inference.add_argument("--video_interval",
                               type=int,
                               default=1,
                               help='num_id to inference')
        inference.add_argument("--num_id",
                               type=int,
                               default=50,
                               help='num_id to inference')

        inference.add_argument(
            "--azim_video",
            action="store_true",
            help=
            "when true, the camera trajectory will travel along the azimuth direction. Otherwise, the camera will travel along an ellipsoid trajectory."
        )
        inference.add_argument(
            "--save_img",
            action="store_true",
            help="when true, the inferenced images will be savged on the disk."
        )
        inference.add_argument(
            "--save_independent_img",
            action="store_true",
            help="when true, the inferenced images will be savged on the disk."
        )
        inference.add_argument("--eval_mode",
                               type=str,
                               default='val',
                               help="when true, use test dataset")

        # Generator options
        model = self.parser.add_argument_group('model')
        model.add_argument("--is_test", action='store_true', help="")
        model.add_argument(
            "--D_input_size",
            type=int,
            default=3,
            help=
            "Discriminator input image size. if 6, concat aligned residual to D to improve residual synthesis."
        )
        model.add_argument("--size",
                           type=int,
                           default=1024,
                           help="image sizes for the model")
        model.add_argument("--style_dim",
                           type=int,
                           default=256,
                           help="number of style input dimensions")
        model.add_argument("--D_init_size",
                           type=int,
                           default=1024,
                           help="initialized size of discriminator")
        model.add_argument(
            "--channel_multiplier",
            type=int,
            default=2,
            help=
            "channel multiplier factor for the StyleGAN decoder. config-f = 2, else = 1"
        )
        model.add_argument(
            "--n_mlp",
            type=int,
            default=8,
            help="number of mlp layers in stylegan's mapping network")
        model.add_argument(
            "--lr_mapping",
            type=float,
            default=0.01,
            help='learning rate reduction for mapping network MLP layers')
        model.add_argument(
            "--renderer_spatial_output_dim",
            type=int,
            default=64,
            help='spatial resolution of the StyleGAN decoder inputs')
        model.add_argument(
            "--project_noise",
            action='store_true',
            help=
            'when true, use geometry-aware noise projection to reduce flickering effects (see supplementary section C.1 in the paper). warning: processing time significantly increases with this flag to ~20 minutes per video.'
        )

        # Camera options
        camera = self.parser.add_argument_group('camera')
        camera.add_argument(
            "--uniform",
            action="store_true",
            help=
            "when true, the camera position is sampled from uniform distribution. Gaussian distribution is the default"
        )
        camera.add_argument("--azim_mean",
                            type=float,
                            default=0.,
                            help="camera azimuth angle std/range in Radians")
        camera.add_argument("--elev_mean",
                            type=float,
                            default=0.,
                            help="camera elevation angle std/range in Radians")
        camera.add_argument("--azim",
                            type=float,
                            default=0.3,
                            help="camera azimuth angle std/range in Radians")
        camera.add_argument("--elev",
                            type=float,
                            default=0.15,
                            help="camera elevation angle std/range in Radians")
        camera.add_argument("--fov",
                            type=float,
                            default=6,
                            help="camera field of view half angle in Degrees")
        camera.add_argument(
            "--dist_radius",
            type=float,
            default=0.12,
            help=
            "radius of points sampling distance from the origin. determines the near and far fields"
        )

        # Volume Renderer options
        training.add_argument(
            "--fixedD",
            action="store_true",
            help="when true, only use pre-trained D as supervisions, no D step"
        )
        training.add_argument(
            "--adaptive_D_loss",
            action="store_true",
            help="when true, use VQGAN adaptive D loss trick.")
        training.add_argument(
            "--evaluate_in_train",
            action="store_true",
            help=
            "when true, the generator will render indentities from a fixed set of camera angles."
        )
        training.add_argument("--w_space",
                              action="store_true",
                              help="use w-space for encoder")

        rendering = self.parser.add_argument_group('rendering')
        # local model
        rendering.add_argument(
            # "--not_use_integrated_surface_normal",
            "--use_integrated_surface_normal",
            action="store_true",
            help=
            "whether to use integrated version of surface normal for the supervision"
        )

        # * merge the following
        # rendering.add_argument(
        #     "--use_L_geo",
        #     action="store_true",
        #     help="only use local model for tex prdiction")
        # rendering.add_argument(
        #     "--use_G_geo",
        #     action="store_true",
        #     help="only use local model for tex prdiction")

        rendering.add_argument(
            "--geo_predictition_strategy",
            type=str,
            default='global',
            choices=['global', 'local', 'global_local'],
            help="only use local branch for all the training, ablation study")

        # merge the following
        # rendering.add_argument(
        #     "--disable_global_model",
        #     action="store_true",
        #     help="only use local branch for all the training, ablation study")

        rendering.add_argument("--tex_predictition_strategy",
                               type=str,
                               default='global',
                               choices=['global', 'local', 'global_local'],
                               help="use what informaiotn to predict texture")

        rendering.add_argument(
            "--local_z_condition",
            action="store_true",
            help="add z condition to Local branch(with siren)")
        rendering.add_argument("--local_offset_norm",
                               action="store_true",
                               help="regularize local offset norm")
        rendering.add_argument("--local_regularize_feats_norm",
                               action="store_true",
                               help="regularize local feats norm")
        rendering.add_argument("--l_geo_residual",
                               action="store_true",
                               help="output sdf residual")
        rendering.add_argument("--use_L_geo_as_residual",
                               action="store_true",
                               help="sdf += residual")
        rendering.add_argument("--L_geo_as_residual_reg",
                               action="store_true",
                               help="residual l2 norm")
        # * more of local branch ablations
        rendering.add_argument("--L_pred_tex_modulations",
                               action="store_true",
                               help="predict modulations rather than features")
        rendering.add_argument("--L_pred_geo_modulations",
                               action="store_true",
                               help="predict modulations rather than features")

        rendering.add_argument("--netLocal_type",
                               type=str,
                               default='HGPIFuNetGAN',
                               help="which pifu model to use")

        rendering.add_argument("--return_feats_layers",
                               nargs='+',
                               type=int,
                               default=[1, 3, 5, 7],
                               help="which layer feature to return")

        rendering.add_argument(
            "--residual_context_feats",
            nargs='+',
            default=['depth'],
            help=
            "what information serves as context information for residual branch"
        )

        rendering.add_argument(
            "--residual_local_feats_dim",
            type=int,
            default=256 + 45,
            help="dimention of residual ocal features -> modulations")

        rendering.add_argument(
            "--residual_PE_type",
            type=str,
            default='coordinate',
            choices=['coordinate', 'depth', 'None'],
            help="use what as the 3D condition, depth or coordinate")
        rendering.add_argument(
            "--local_append_E_feature_map",
            action='store_true',
            help=
            "append feature map from encoder as the input to the residual netLocal encoder"
        )
        rendering.add_argument(
            "--local_modulation_layer",
            #    type=int,
            #    default=-1,
            action='store_true',
            help="predict modulations rather than features")
        rendering.add_argument(
            "--local_modulation_layer_in_backbone",
            #    type=int,
            #    default=-1,
            action='store_true',
            help="old behaviour, for ablations")
        rendering.add_argument(
            "--local_modulation_layer_in_backbone_afterViewLayer",
            #    type=int,
            #    default=-1,
            action='store_true',
            help="old behaviour, for ablations")
        # rendering.add_argument(
        #     "--use_L_geo",
        #     action="store_true",
        #     help="only use local model for tex prdiction")
        # rendering.add_argument(
        #     "--use_G_geo",
        #     action="store_true",
        #     help="only use local model for tex prdiction")
        rendering.add_argument(
            "--enable_local_model",
            action="store_true",
            help="train local-prior and global-prior networks only")
        # sampling
        rendering.add_argument(
            "--add_fg_mask",  # todo
            action="store_true",
            help="use weights and biases logging")
        training.add_argument('--enable_custom_grid_sample',
                              action='store_true',
                              help='use nvidia op')
        training.add_argument("--disable_decoder_fpn",
                              action="store_true",
                              help="don't create decoder fpn")
        training.add_argument("--fg_mask",
                              action="store_true",
                              help="use weights and biases logging")

        rendering.add_argument("--sample_near_surface",
                               action="store_true",
                               help="for near surface points sampling")

        rendering.add_argument("--sample_uniform_grid",
                               action="store_true",
                               help="sample uniform in scene grid")

        # 3d supervision sampling
        rendering.add_argument(
            "--uniform_grid_sampling_num",
            type=int,
            default=2048,  # follow deepsdf
            help="for uniform space points sampling")
        rendering.add_argument(
            "--near_surface_sampling_num",
            type=int,
            default=4096 * 2,  # follow deepsdf
            help="for near surface points sampling")
        rendering.add_argument(
            "--surface_sampling_stdv",
            type=float,
            default=0.03,  # 4 sigma covers the whole scene, 95.44
            help="for near surface points sampling")

        rendering.add_argument("--surface_add_randn_noise",
                               action="store_true",
                               help="")
        # DDF
        rendering.add_argument("--uniform_from_surface",
                               action="store_true",
                               help="use weights and biases logging")
        # DDF
        rendering.add_argument("--ddf",
                               action='store_true',
                               help='use ddf model')
        # MLP model parameters
        rendering.add_argument("--depth",
                               type=int,
                               default=8,
                               help='layers in network')
        rendering.add_argument("--width",
                               type=int,
                               default=256,
                               help='channels per layer')
        # surface sampling options
        rendering.add_argument("--surface_sampling_num",
                               type=int,
                               default=5000,
                               help='points random sampled to query surface')
        # Volume representation options
        rendering.add_argument(
            "--no_sdf",
            action='store_true',
            help=
            'By default, the raw MLP outputs represent an underline signed distance field (SDF). When true, the MLP outputs represent the traditional NeRF density field.'
        )
        rendering.add_argument(
            "--no_z_normalize",
            action='store_true',
            help=
            'By default, the model normalizes input coordinates such that the z coordinate is in [-1,1]. When true that feature is disabled.'
        )
        rendering.add_argument(
            "--spatial_super_sampling_factor",
            type=int,
            default=1,
            help='for super sampling the original 64 resolution.')
        rendering.add_argument(
            "--static_viewdirs",
            action='store_true',
            help='when true, use static viewing direction input to the MLP')
        # Ray intergration options
        rendering.add_argument("--uniform_surface_ratio",
                               type=float,
                               default=0.25,
                               help='number of samples per ray')
        rendering.add_argument("--N_samples",
                               type=int,
                               default=24,
                               help='number of samples per ray')
        rendering.add_argument(
            "--no_offset_sampling",
            action='store_true',
            help=
            'when true, use random stratified sampling when rendering the volume, otherwise offset sampling is used. (See Equation (3) in Sec. 3.2 of the paper)'
        )
        rendering.add_argument("--perturb",
                               type=float,
                               default=1.,
                               help='set to 0. for no jitter, 1. for jitter')
        rendering.add_argument(
            "--raw_noise_std",
            type=float,
            default=0.,
            help=
            'std dev of noise added to regularize sigma_a output, 1e0 recommended'
        )
        rendering.add_argument(
            "--force_background",
            action='store_true',
            help=
            'force the last depth sample to act as background in case of a transparent ray'
        )
        rendering.add_argument("--return_feats",
                               action='store_true',
                               help='for vis and debugging')
        rendering.add_argument(
            "--disable_ref_view_mask",
            action='store_true',
            help='do not use the ref view hit_prob to weight the local features'
        )
        rendering.add_argument(
            "--disable_ref_view_weight",
            action='store_true',
            help='do not use the ref view hit_prob to weight the local features'
        )
        # rendering.add_argument(
        #     "--not_apply_que_weight",
        #     action='store_true',
        #     help='only use the ref view hit_prob to weight the local features')
        # Set volume renderer outputs
        rendering.add_argument(
            "--return_xyz",
            action='store_true',
            help=
            'when true, the volume renderer also returns the xyz point could of the surface. This point cloud is used to produce depth map renderings'
        )
        rendering.add_argument(
            "--return_sdf",
            action='store_true',
            help=
            'when true, the volume renderer also returns the SDF network outputs for each location in the volume'
        )
        # for 3d reconstruction
        training.add_argument(
            '--return_surface_eikonal',
            action='store_true',
            help='calculate surface normal for shape reconstruction')
        training.add_argument('--test_real_on_synthetic',
                              action='store_true',
                              help='ignore pixel supervision')
        training.add_argument('--no_pix_sup',
                              action='store_true',
                              help='ignore pixel supervision')
        training.add_argument('--pix_sup_only',
                              action='store_true',
                              help='ignore pixel supervision')
        training.add_argument('--shape_sup_only',
                              action='store_true',
                              help='ignore pixel supervision')

        # for auto-encoder
        training.add_argument('--runner',
                              default='AERunner',
                              type=str,
                              help='which runner to call, for mmcv registry')
        training.add_argument('--overfitting',
                              action='store_true',
                              help='overfitting on sampled identitied')
        training.add_argument('--analyze_w_mixing',
                              action='store_true',
                              help='which encoder coach to use')
        training.add_argument('--coach',
                              default='e4e',
                              type=str,
                              help='which encoder coach to use')
        training.add_argument("--full_pipeline",
                              action="store_true",
                              help="train 2 G?")

        experiment.add_argument("--w_space_style_pred",
                                action="store_true",
                                help="duplicate w-mean if set to true")
        training.add_argument("--fg_threshold",
                              type=float,
                              default=1.10,
                              help="remove bg supervision")
        training.add_argument("--latent_reg",
                              type=float,
                              default=0,
                              help="weight of the r1 regularization")
        training.add_argument(
            "--curriculum_pose_sampling_iternum",
            type=float,
            default=-1,
            help="linear increate the std of the sampled pose.")
        training.add_argument("--real_lambda",
                              type=float,
                              default=1,
                              help="weight of the real part training")
        training.add_argument("--id_lambda",
                              type=float,
                              default=0.1,
                              help="weight of the identity loss")
        training.add_argument("--disc_lambda",
                              type=float,
                              default=0.,
                              help="weight of the discriminator loss")
        training.add_argument("--vgg_lambda",
                              type=float,
                              default=0.8,
                              help="weight of the vgg loss")
        training.add_argument("--rec_loss",
                              type=str,
                              default='l2',
                              help='results/evaluations directory name')
        # training.add_argument("--disable_pix",
        #                       type=str,
        #                       default='l2',
        #                       help='results/evaluations directory name')
        model.add_argument("--encoder_ckpt",
                           type=str,
                           default='',
                           help='pretrained encoder')

        model.add_argument("--renderer_encoder_ckpt",
                           type=str,
                           default='',
                           help='pretrained encoder')
        model.add_argument("--stylegan_encoder_ckpt",
                           type=str,
                           default='',
                           help='pretrained encoder')
        model.add_argument("--full_encoder_ckpt",
                           type=str,
                           default='',
                           help='pretrained encoder')
        training.add_argument("--train_",
                              action="store_true",
                              help="2 encoders")
        training.add_argument("--enable_G1_only",
                              action="store_true",
                              help="train decoder")
        training.add_argument("--supervise_both_gen_imgs",
                              action="store_true",
                              help="supervise output from 2 encoders")
        training.add_argument(
            "--train_regularize_mapping",  # todo
            action="store_true",
            help="reg later with current mapping")
        # arguments for iterative encoding
        # encoder = training.add_argument_group('encoder')
        training.add_argument(
            '--n_iters_per_batch',
            default=5,
            type=int,
            help='number of forward passes per batch during training')
        training.add_argument('--encoder_type',
                              default='HybridGradualStyleEncoder_V2',
                              type=str,
                              help='which encoder to use')
        training.add_argument(
            '--input_nc',
            default=3,
            type=int,
            help=
            'number of input image channels to the restyle encoder. should be set to 6.'
        )
        training.add_argument('--encoder_input_size',
                              default=256,
                              type=int,
                              help='output size of generator')
        training.add_argument('--output_size',
                              default=256,
                              type=int,
                              help='output size of generator')
        training.add_argument('--dataset_type',
                              default='ffhq_encode',
                              type=str,
                              help='type of dataset/experiment to run')
        # self.encoder.add_argument('--dataset_type', default='ffhq_encode', type=str, help='type of dataset/experiment to run')

        # encoder-optimizers
        # training.add_argument('--learning_rate',
        #                       default=0.0001,
        #                       type=float,
        #                       help='optimizer learning rate')
        training.add_argument('--optim_name',
                              default='adam',
                              type=str,
                              help='which optimizer to use')
        training.add_argument('--enable_G1',
                              action='store_true',
                              help='whether to train the decoder model')
        training.add_argument(
            '--start_from_latent_avg',
            action='store_true',
            help=
            'whether to add average latent vector to generate codes from encoder.'
        )
        training.add_argument(
            '--volume_discriminator_path',
            default='pretrained_renderer/ffhq_vol_renderer.pt',
            type=str,
            help='path to pretrained volume D model weights')
        training.add_argument('--Discriminator_ckpt_path',
                              default=None,
                              type=str,
                              help='path to pretrained G model weights')
        training.add_argument('--stylesdf_weights',
                              default=None,
                              type=str,
                              help='path to pretrained G model weights')
        training.add_argument('--exp_dir',
                              type=str,
                              help='path to experiment output directory')
        training.add_argument('--synthetic_batch_size',
                              default=2,
                              type=int,
                              help='batch size for training')
        # training.add_argument('--batch_size',
        #                       default=4,
        #                       type=int,
        #                       help='batch size for training')
        training.add_argument('--test_batch_size',
                              default=2,
                              type=int,
                              help='batch size for testing and inference')
        training.add_argument(
            '--testset_size',
            default=50,
            type=int,
            help='testset size (of randn latents) for testing and inference')
        training.add_argument('--workers',
                              default=2,
                              type=int,
                              help='number of train dataloader workers')
        training.add_argument(
            '--test_workers',
            default=2,
            type=int,
            help='number of test/inference dataloader workers')

        # loss lambdas
        training.add_argument('--mae_lambda',
                              default=0.,
                              type=float,
                              help='l1 loss multiplier factor')
        training.add_argument('--ssim_lambda',
                              default=0.,
                              type=float,
                              help='ssim loss multiplier factor')
        training.add_argument('--lpips_lambda',
                              default=0.8,
                              type=float,
                              help='lpips loss multiplier factor')
        training.add_argument('--lpips_type',
                              default='alex',
                              type=str,
                              help='LPIPS backbone')

        # training.add_argument('--id_lambda', default=0, type=float, help='id loss multiplier factor')
        training.add_argument('--lms_lambda',
                              default=0,
                              type=float,
                              help='facial landmarks loss multiplier factor')
        training.add_argument('--l2_lambda',
                              default=1,
                              type=float,
                              help='l2 loss multiplier factor')
        training.add_argument('--w_norm_lambda',
                              default=0,
                              type=float,
                              help='w-norm loss multiplier factor')
        training.add_argument('--moco_lambda',
                              default=0,
                              type=float,
                              help='moco feature loss multiplier factor')
        training.add_argument('--w_discriminator_lambda',
                              default=0,
                              type=float,
                              help='img discriminator lambda')
        training.add_argument('--w_latent_discriminator_lambda',
                              default=0,
                              type=float,
                              help='latent discriminator lambda')
        # training.add_argument('--res_lambda', default=0., type=float,
        #                       help='(res-res_gt) multiplier factor')

        # log
        training.add_argument('--max_steps',
                              default=500000,
                              type=int,
                              help='maximum number of training steps')
        training.add_argument(
            '--image_interval',
            default=100,
            type=int,
            help='interval for logging train images during training')
        training.add_argument(
            '--board_interval',
            default=50,
            type=int,
            help='interval for logging metrics to tensorboard')
        training.add_argument('--start_iter',
                              default=-1,
                              type=int,
                              help='manually assign when using pt model')
        training.add_argument('--seed',
                              default=0,
                              type=int,
                              help='save ckpt interval')
        training.add_argument('--ckpt_interval',
                              default=10000,
                              type=int,
                              help='save ckpt interval')
        training.add_argument('--saveimg_interval',
                              default=100,
                              type=int,
                              help='save train img interval')
        training.add_argument('--val_interval',
                              default=1000,
                              type=int,
                              help='validation interval')
        training.add_argument('--save_interval',
                              default=5000,
                              type=int,
                              help='model checkpoint interval')
        training.add_argument('--checkpoint_path',
                              default=None,
                              type=str,
                              help='path to restyle model checkpoint')
        training.add_argument('--no_init_encoder',
                              action='store_true',
                              help='whether to init encoder.')
        # training.add_argument('--fix_renderer_encoder',
        #                       action='store_true',
        #                       help='whether to fix encoder.')

        # training.add_argument('--fix_renderer_encoder',
        #                       action='store_true',
        #                       help='whether to fix pretrained encoder.')

        # Discriminator flags
        # training.add_argument('--w_discriminator_lambda', default=0, type=float, help='Dw loss multiplier')
        training.add_argument('--w_discriminator_lr',
                              default=2e-5,
                              type=float,
                              help='Dw learning rate')
        # training.add_argument("--r1", type=float, default=10, help="weight of the r1 regularization")
        # training.add_argument("--d_reg_every", type=int, default=16,
        #                          help="interval for applying r1 regularization")
        training.add_argument(
            '--use_w_pool',
            action='store_true',
            help=
            'Whether to store a latnet codes pool for the discriminator\'s training'
        )
        training.add_argument("--w_pool_size",
                              type=int,
                              default=50,
                              help="W\'s pool size, depends on --use_w_pool")

        # e4e specific
        training.add_argument('--delta_norm',
                              type=int,
                              default=2,
                              help="norm type of the deltas")
        training.add_argument('--delta_norm_lambda',
                              type=float,
                              default=2e-4,
                              help="lambda for delta norm loss")

        # Progressive training
        training.add_argument(
            '--progressive_pose_sampling',
            action='store_true',
            help="whether to enable progressive pose sampling strategy")
        training.add_argument(
            '--progressive_pose_lambda',
            nargs='+',
            type=float,
            default=[0, 0.15, 0.25, 0.5, 0.75, 1],
            help="The weights of cross-view pose std sampling; ")
        training.add_argument(
            '--progressive_pose_steps',
            nargs='+',
            type=int,
            # default=[0, 20000, 24000, 28000, 32000, 36000],
            default=[0, 10000, 14000, 18000, 22000, 26000],
            help=
            "The training steps of each pose sampling range, after the last steps the lamdba should be 1."
        )
        training.add_argument(
            '--progressive_steps',
            nargs='+',
            type=int,
            default=None,
            help=
            "The training steps of training new deltas. steps[i] starts the delta_i training"
        )
        training.add_argument(
            '--progressive_start',
            type=int,
            default=None,
            help=
            "The training step to start training the deltas, overrides progressive_steps"
        )
        training.add_argument(
            '--progressive_step_every',
            type=int,
            default=2_000,
            help="Amount of training steps for each progressive step")

        # Save additional training info to enable future training continuation from produced checkpoints
        training.add_argument(
            '--save_training_data',
            action='store_true',
            help=
            'Save intermediate training data to resume training from the checkpoint'
        )
        training.add_argument('--sub_exp_dir',
                              default=None,
                              type=str,
                              help='Name of sub experiment directory')
        training.add_argument(
            '--keep_optimizer',
            action='store_true',
            help='Whether to continue from the checkpoint\'s optimizer')
        training.add_argument(
            '--resume_training_from_ckpt',
            default=None,
            type=str,
            help=
            'Path to training checkpoint, works when --save_training_data was set to True'
        )

        training.add_argument(
            '--update_param_list',
            nargs='+',
            type=str,
            default=None,
            help=
            "Name of training parameters to update the loaded training checkpoint"
        )

        # * for training steps
        training.add_argument('--cycle_training',
                              action='store_true',
                              help='Whether to adopt swap-code cycle training')

        training.add_argument('--hit_prob_consistency_lambda',
                              default=0.1,
                              type=float,
                              help='(res-res_gt) multiplier factor')
        training.add_argument('--depth_consistency_lambda',
                              default=0.1,
                              type=float,
                              help='(res-res_gt) multiplier factor')
        training.add_argument('--res_lambda_thumb',
                              default=0.,
                              type=float,
                              help='(res-res_gt) multiplier factor')
        training.add_argument('--res_lambda',
                              default=0.1,
                              type=float,
                              help='(res-res_gt) multiplier factor')
        training.add_argument('--feat_sim_loss_2d',
                              default=0,
                              type=float,
                              help='weight for 2D feature map sim loss')
        training.add_argument(
            '--corr_grid_lambda',
            default=1,
            type=float,
            help='weight for predicted correspondence grid map')

        training.add_argument(
            '--latent_gt_lambda',
            default=0.,
            type=float,
            help='compare latent code with the sampled W space gt latents')
        training.add_argument(
            '--latent_cycle_lambda',
            default=0.,
            type=float,
            help='latent code from different pose multiplier factor')
        training.add_argument('--distortion_scale',
                              type=float,
                              default=0.15,
                              help="lambda for delta norm loss")
        training.add_argument('--aug_rate',
                              type=float,
                              default=0.8,
                              help="lambda for delta norm loss")
        training.add_argument(
            '--train_discriminator_step_interval',
            default=1,
            type=int,
            help='train D every N steps, avoid D overfitting')
        training.add_argument('--adv_lambda',
                              default=0,
                              type=float,
                              help='adversarial loss multiplier')
        training.add_argument('--discriminator_lambda',
                              default=0,
                              type=float,
                              help='Dw loss multiplier')
        training.add_argument('--discriminator_lr',
                              default=2e-5,
                              type=float,
                              help='Dw learning rate')
        # for fpn
        training.add_argument('--pigan_geo_layer',
                              default=6,
                              type=int,
                              help='split 3D renderer geo/tex layers')
        training.add_argument('--pigan_tex_layer',
                              default=9,
                              type=int,
                              help='split 3D renderer geo/tex layers')
        training.add_argument("--fpn_pigan_geo_layer_dim",
                              type=int,
                              default=128)
        training.add_argument("--fpn_pigan_tex_layer_dim",
                              type=int,
                              default=128)
        training.add_argument("--fpn_pigan_stylegan_layer_dim",
                              type=int,
                              default=128)
        training.add_argument(
            '--single_decoder_layer',
            action='store_true',
            help='Whether to continue from the checkpoint\'s optimizer')
        training.add_argument("--ckpt_to_ignore",
                              nargs="*",
                              default="",
                              help="help fix module")

        sampling = self.parser.add_argument_group('sampling')
        sampling.add_argument('--sampling_mode',
                              type=str,
                              default='uniform',
                              help='uniform or density importance sampling')
        sampling.add_argument('--sample_rays',
                              type=int,
                              default=32 * 32,
                              help='how many rays to use in a batch')
        sampling.add_argument('--foreground_sampling',
                              action='store_true',
                              help='remove bg rays')

        # editing
        editing = self.parser.add_argument_group('editing')
        editing.add_argument(
            '--boundary_path',
            type=str,
            default='/mnt/lustre/yslan/Repo/Research/SIGA22/interfacegan')
        # editing.add_argument('--projection_logging_interval', type=int, default=35)
        editing.add_argument('--space', type=str, default='renderer')
        editing.add_argument('--render_video_for_each_direction',
                             action='store_true')

        # modified from https://github.dev/danielroich/PTI
        projection = self.parser.add_argument_group('projection')
        projection.add_argument('--PTI', action='store_true', help='ft G')
        projection.add_argument('--reverse_file_order',
                                action='store_true',
                                help='ft G')
        projection.add_argument('--inversion_calc_metrics',
                                action='store_true',
                                help='calculate metrics for inversion')
        projection.add_argument("--inverse_files",
                                metavar="FILES",
                                nargs="+",
                                help="path to image files to be projected")
        projection.add_argument('--wspace',
                                action='store_true',
                                help='w space to inverse.')

        projection.add_argument('--projection_logging_interval',
                                type=int,
                                default=150)
        projection.add_argument('--pt_l2_lambda', type=float, default=1)
        projection.add_argument('--pt_lpips_lambda', type=float, default=1)
        projection.add_argument('--LPIPS_value_threshold',
                                type=float,
                                default=0.06)

        projection.add_argument('--max_pti_steps', type=int, default=100)
        projection.add_argument('--first_inv_steps', type=int, default=300)
        projection.add_argument('--w_inversion_root',
                                type=str,
                                help='w space code directory inversed.')
        projection.add_argument('--max_images_to_invert', type=int, default=30)
        projection.add_argument('--first_inv_lr', type=float, default=5e-3)
        projection.add_argument('--pti_learning_rate',
                                type=float,
                                default=5e-5)

        projection.add_argument('--inference_projection_validation',
                                action='store_true',
                                help='')
        # projection.add_argument('--pti_inference', action='store_true', help='')

        self.initialized = True

    # todo, write a local file loader
    def parse(self, filter_key=None, args=None):
        """parse args

        Args:
            filter_key (list, optional): filter_key used to create separate naming space for parsers merged. Defaults to None.

        Returns:
            Munch: args in dotmap
        """
        # st()
        if self.opt is None:
            self.opt = Munch()
        if not self.initialized:
            self.initialize()

        try:
            args = self.parser.parse_args(args=args)
        except:  # solves argparse error in google colab
            args = self.parser.parse_args(args=[])

        if filter_key is not None:
            self.opt[filter_key] = Munch()

        for group in self.parser._action_groups[2:]:
            title = group.title
            # print(title)
            if filter_key is None or filter_key not in title:
                self.opt[title] = Munch()
            else:  # parsers to merge into seperace unified naming space filter-key
                title = filter_key

            for action in group._group_actions:
                dest = action.dest
                self.opt[title][dest] = args.__getattribute__(dest)

        return self.opt
