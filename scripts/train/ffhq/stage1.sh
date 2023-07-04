set -x

batch_size=4 # batch size per GPU
ngpu=4 # set your GPU number here

expname=ffhq1024x1024

dataset_path=/mnt/lustre/share/shuaiyang/ffhq/realign256x256/
eval_dataset_path=assets/demo_imgs 
test_dataset_path=datasets/test_img 

version=1 # experiment version, for log
exp_prefix=version_${version}_gpu${ngpu}
checkpoints_dir=logs/train/ffhq/stage1/${exp_prefix}

encoder_type=HybridGradualStyleEncoder_V2 # a modified version of pSp network for 3D GANs

python -m torch.distributed.launch \
--master_port 22008 \
--nproc_per_node $ngpu train_ae.py \
--checkpoints_dir $checkpoints_dir \
--batch $batch_size \
--synthetic_batch_size $batch_size \
--chunk $batch_size \
--dataset_path $dataset_path \
--eval_dataset_path $eval_dataset_path \
--test_dataset_path ${test_dataset_path} \
--expname $expname \
--size 1024 \
--return_xyz \
--full_pipeline \
--w_space_style_pred \
--fpn_pigan_geo_layer_dim 128 \
--encoder_type ${encoder_type} \
--val_n_sample 1 \
--id_lambda 0.1 \
--vgg_lambda 0.8 \
--lr 5e-5 \
--return_xyz \
--synthetic_sampling_strategy all_fake \
--l2_lambda 1 \
--vgg_lambda 0.8 \
--surf_normal_lambda 1 \
--surf_sdf_lambda 1 \
--uniform_pts_sdf_lambda 0.2 \
--eikonal_lambda 0.1 \
--return_surface_eikonal \
--val_interval 2000 \
--fg_mask \
--N_samples 18 \
--sample_uniform_grid \
--sample_near_surface \
--E_d_grad_false \
--E_l_grad_false \
--latent_gt_lambda 1 \
--eval_batch 1 \
--start_iter 0 \
--disable_decoder_fpn \
--wandb \