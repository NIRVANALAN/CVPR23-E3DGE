set -x
# stage 2.1, train netLocal with 3D projected feature (using 2D alignment module)

batch_size=8 # batch size per GPU
expname=ffhq1024x1024
ngpu=4

version=1 # modify yourself
exp_prefix=version${version}_gpu${ngpu}
checkpoints_dir=logs/train/ffhq/stage2.1/${exp_prefix} # experiment save path

eval_dataset_path=assets/demo_imgs 
test_dataset_path=datasets/test_img 
dataset_path=${eval_dataset_path} # the FFHQ/AFHQ/ShapeNet dataset path, though it is not used in this stage

# what runner to use
runner=E3DGE_2DAlignOnly_Runner
hg_input_channel=64
netLocal_type=HGPIFuNetGANResidual # a modified version of pifu Hourglass model

ckpt_path='' # put your stage 1 checkpoint .pt path here

python -m torch.distributed.launch \
--master_port 22003 \
--nproc_per_node $ngpu train_ae.py \
--checkpoints_dir $checkpoints_dir \
--synthetic_batch_size $synthetic_batch_size \
--chunk $chunk \
--dataset_path ${dataset_path} \
--eval_dataset_path $eval_dataset_path \
--test_dataset_path ${test_dataset_path} \
--expname $expname \
--size 1024 \
    --full_pipeline \
    --val_n_sample 1 \
    --no_surface_renderings \
    --return_xyz \
    --force_background \
--synthetic_sampling_strategy all_fake \
--val_interval 2000 \
--fg_mask \
--N_samples 24 \
--cycle_training \
    --E_backbone_false \
    --E_g_grad_false \
    --E_d_grad_false \
    --w_space_style_pred \
  --fpn_pigan_geo_layer_dim 128 \
  --encoder_type HybridGradualStyleEncoder_V2 \
  --enable_local_model \
  --L_pred_tex_modulations \
  --tex_predictition_strategy global_local \
  --residual_context_feats depth \
  --residual_PE_type coordinate \
  --local_modulation_layer \
  --hg_input_channel $hg_input_channel \
  --netLocal_type $netLocal_type \
  --runner $runner \
      --lr 5e-5 \
      --pix_sup_only \
      --lambda_l 1 \
      --vgg_lambda 0.8 \
      --id_lambda 0.1 \
      --lpips_lambda 0.8 \
      --l2_lambda 1 \
      --surf_normal_lambda 0 \
      --surf_sdf_lambda 0 \
      --uniform_pts_sdf_lambda 0. \
      --discriminator_lambda 0 \
      --adv_lambda 0 \
      --input_nc 3 \
      --enable_G1 \
      --ckpt ${ckpt_path} \
      --res_lambda_thumb 0.1 \
      --res_lambda 1 \
      --ckpt_to_ignore netLocal grid_align \
      --progressive_pose_sampling \
      --supervise_both_gen_imgs \
      --progressive_pose_steps 434000 454900 465050 476300 487450 494000 \
      --perturb 0 \
      --wandb \
      # --fix_ada \
      # --progressive_pose_steps 434000 434900 435050 436300 437450 444000 \

    #   --fix_renderer
    #   --progressive_pose_steps 0 5000 6250 7500 8650 10000 \