set -x

runner=E3DGE_Full_Runner

# test_dataset_path=/mnt/lustre/yslan/Repo/3D/correspondence/cvpr22/CelebAMask-HQ/face_parsing/Data_preprocessing/test_img 
# test_dataset_path=assets/demo_imgs
test_dataset_path=assets/colab_demo_img
eval_dataset_path=${test_dataset_path}
dataset_path=${test_dataset_path}

encoder_type=HybridGradualStyleEncoder_V2

# to render surface:
batch_size=1
synthetic_batch_size=1
chunk=1
eval_batch=1

# runner=occlusion_runner_cycle_v9_hybridAlign
# ckpt_weight=0.01
ckpt_path=pretrained_models/E3DGE_Full_Runner.pt
weight=0.01

checkpoints_dir=logs/editing/

export CUDA_VISIBLE_DEVICES=0

python test_ae.py \
--eval_dataset_path $eval_dataset_path \
--dataset_path $dataset_path \
--test_dataset_path ${test_dataset_path} \
--checkpoints_dir $checkpoints_dir \
--synthetic_batch_size $synthetic_batch_size \
--eval_batch $eval_batch \
--batch $batch_size \
--chunk $chunk \
--expname ffhq1024x1024 \
--size 1024 \
--full_pipeline \
--w_space_style_pred \
--fpn_pigan_geo_layer_dim 128 \
--encoder_type $encoder_type \
--val_n_sample 1 \
--lr 5e-4 \
--fg_mask \
--continue_training \
--loadSize 256 \
--z_size 1.12 \
--freq_eval 5000 \
--freq_save 5000 \
--local_prior --random_flip --random_scale --random_trans \
--val_interval 2000 \
--full_pipeline \
--pix_sup_only \
--lambda_l 1 \
--vgg_lambda 1 \
--id_lambda 10 \
--lpips_lambda 2 \
--l2_lambda 1 \
--tex_predictition_strategy global_local \
--synthetic_sampling_strategy all_fake \
--surf_normal_lambda 0 \
--surf_sdf_lambda 0 \
--uniform_pts_sdf_lambda 0. \
--iter 300000 \
--enable_local_model \
--image_interval 100 \
 --enable_G1 \
 --E_backbone_false \
 --E_g_grad_false \
 --E_d_grad_false \
 --seed 2 \
 --cycle_training \
 --swap_res \
 --local_modulation_layer \
 --L_pred_tex_modulations \
 --runner $runner \
 --return_xyz \
 --force_background \
 --not_apply_que_weight \
 --ckpt $ckpt_path \
 --local_append_E_feature_map \
 --hg_input_channel 64 \
 --netLocal_type HGPIFuNetGANResidualResnetFC \
 --residual_context_feats depth $feat_maps \
 --residual_PE_type coordinate \
 --residual_local_feats_dim 301 \
 --res_lambda 0.1 \
 --progressive_pose_sampling \
 --view_lambda 0 \
 --discriminator_lambda 0 \
 --adv_lambda 0 \
 --D_init_size 256 \
 --save_independent_img \
--input_nc 3 \
--disable_ref_view_weight \
--editing_inference \
--azim_video \
--video_frames 8 \
--editing_boundary_dir assets/editing_boundaries/stylesdf \
--render_video_for_each_direction \
--no_surface_renderings \
# --editing_boundary_scale_upperbound 0 0 1 0 0 \
# --editing_boundary_scale_lowerbound 0 0 1 0 0 \