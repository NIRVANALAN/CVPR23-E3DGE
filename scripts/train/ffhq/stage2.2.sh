set -x

# stage 2.2, train netLocal with 3D projected feature

batch_size=4
synthetic_batch_size=${batch_size}
chunk=${batch_size}
r1=60
res_lambda=0.1
id_lambda=0.1

expname=ffhq1024x1024

ngpu=4
# ngpu=1

version=1 # modify yourself
exp_prefix=version${version}_gpu${ngpu}
checkpoints_dir=logs/train/ffhq/stage2.2/${exp_prefix}

eval_dataset_path=assets/demo_imgs 
test_dataset_path=datasets/test_img 
dataset_path='' # the FFHQ/AFHQ/ShapeNet dataset path, used for adversarial training in this stage

adv_lambda=0.01 # set to nonzero if enables adversarial training
# weight=0
runner=E3DGE_Full_Runner
hg_input_channel=64
netLocal_type=HGPIFuNetGANResidual

# ! set the stage 2.1 pre-trained checkpoint path here: 
ckpt_path=''

python -m torch.distributed.launch \
--master_port 22003 \
--nproc_per_node $ngpu train_ae.py \
--checkpoints_dir $checkpoints_dir \
--synthetic_batch_size $synthetic_batch_size \
--chunk $chunk \
--dataset_path ${dataset_path} \
--eval_dataset_path $eval_dataset_path \
--test_dataset_path ${test_dataset_path} \
--expname ${expname} \
--size 1024 \
--full_pipeline \
--val_n_sample 1 \
--no_surface_renderings \
--return_xyz \
--force_background \
--synthetic_sampling_strategy all_fake \
--val_interval 2000 \
--wandb \
--fg_mask \
--N_samples 24 \
--cycle_training \
--swap_res \
--E_backbone_false \
--E_g_grad_false \
--E_d_grad_false \
--w_space_style_pred \
--fpn_pigan_geo_layer_dim 128 \
--encoder_type HybridGradualStyleEncoder_V2 \
--enable_local_model \
--local_modulation_layer \
--L_pred_tex_modulations \
--tex_predictition_strategy global_local \
--hg_input_channel $hg_input_channel \
--netLocal_type $netLocal_type \
--residual_context_feats depth \
--residual_PE_type coordinate \
--residual_local_feats_dim 301 \
--res_lambda 1 \
--res_lambda_thumb 0.1 \
--supervise_both_gen_imgs \
--view_lambda 0 \
--progressive_pose_sampling \
--disable_ref_view_weight  \
--runner $runner \
--lr 5e-5 \
--pix_sup_only \
--lambda_l 1 \
--vgg_lambda 1 \
--id_lambda ${id_lambda} \
--lpips_lambda 0.8 \
--l2_lambda 1 \
--surf_normal_lambda 0 \
--surf_sdf_lambda 0 \
--uniform_pts_sdf_lambda 0. \
--discriminator_lambda ${adv_lambda} \
--adv_lambda ${adv_lambda} \
--D_init_size 256 \
--input_nc 3 \
--enable_G1 \
--perturb 0 \
--ckpt ${ckpt_path} \
--fix_ada