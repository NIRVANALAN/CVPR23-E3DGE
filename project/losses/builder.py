import kornia
from torch.nn import functional as F
import torch

from . import IDLoss, LPIPS, eikonal_loss


class LossClass(torch.nn.Module):

    def __init__(self, device, opt) -> None:
        super().__init__()

        self.opt = opt
        self.device = device
        # define 2D loss class
        self.criterionImg = torch.nn.MSELoss()
        # if device == 'cpu':
        #     self.criterionLPIPS = torch.nn.Identity()  # for debugging
        # else:
        self.criterionLPIPS = LPIPS(net_type='alex', device=device).eval()

        # self.criterionID = IDLoss().to(device).eval()
        self.criterionID = IDLoss(device=device).eval()

        # define 3d rec loss
        self.criterion3d_rec = torch.nn.SmoothL1Loss()
        self.id_loss_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        self.uniform_pts_loss = torch.nn.BCEWithLogitsLoss()

        # if opt.fix_beta: # not learnable
        #     self.register_buffer('sigmoid_beta', torch.ones(1) * 0.1)
        # else:
        # self.sigmoid_beta = nn.Parameter(0.1*torch.ones(1)) # smaller -> shaper sdf
        # self.sigmoid_beta = nn.Parameter(100*torch.ones(1)) # smaller -> shaper sdf
        # * todo

        print('init loss class finished', flush=True)

    # def psnr_loss(self, input: torch.Tensor, target: torch.Tensor, max_val: float) -> torch.Tensor:
    def psnr_loss(self, input, target, max_val):
        return kornia.metrics.psnr(input, target, max_val)

    def calc_shape_rec_loss(self,
                            pred_shape: dict,
                            gt_shape: dict,
                            device,
                            supervise_sdf=True,
                            supervise_surface=False,
                            supervise_surface_normal=False,
                            supervise_eikonal=False,
                            supervise_chamfer_dist=False):
        """apply 3d shape reconstruction supervision

        Args:
            pred_shape (dict): dict contains reconstructed shape information
            gt_shape (dict): dict contains gt shape information
            supervise_sdf (bool, optional): whether supervise sdf rec. Defaults to True.
            supervise_surface_normal (bool, optional): whether supervise surface rec. Defaults to False.

        Returns:
            dict: shape reconstruction loss
        """

        shape_loss_dict = {}
        shape_loss = 0
        assert supervise_sdf or supervise_surface_normal, 'should at least supervise one types of shape reconstruction'
        # todo, add weights

        shape_loss_dict['surf_rec_loss'] = torch.tensor(0., device=device)
        shape_loss_dict['sdf_rec_loss'] = torch.tensor(0., device=device)
        shape_loss_dict['surface_norm_rec_loss'] = torch.tensor(0.,
                                                                device=device)
        shape_loss_dict['eikonal_term'] = torch.tensor(0., device=device)
        # shape_loss_dict['chamfer_distance'] = torch.tensor(0., device=device)

        if supervise_sdf and self.opt.uniform_pts_sdf_lambda > 0:
            # st()
            # shape_loss_dict['sdf_rec_loss'] = self.uniform_pts_loss(
            #     pred_shape['sdf']*self.sigmoid_beta,
            #     (gt_shape['sdf']>0).float()) * self.opt.uniform_pts_sdf_lambda
            shape_loss_dict['sdf_rec_loss'] = self.criterion3d_rec(
                pred_shape['uniform_points_sdf'].squeeze(),
                gt_shape['uniform_points_sdf'].squeeze(
                )) * self.opt.uniform_pts_sdf_lambda
            shape_loss = shape_loss + shape_loss_dict['sdf_rec_loss']

        if supervise_surface and self.opt.surf_sdf_lambda > 0:
            shape_loss_dict['surf_rec_loss'] = self.criterion3d_rec(
                pred_shape['surface_sdf'],
                torch.zeros_like(
                    pred_shape['surface_sdf'])) * self.opt.surf_sdf_lambda
            shape_loss = shape_loss + shape_loss_dict['surf_rec_loss']

        if supervise_surface_normal and self.opt.surf_normal_lambda > 0:
            shape_loss_dict['surface_norm_rec_loss'] = self.criterion3d_rec(
                pred_shape['surface_eikonal_term'].squeeze(),
                gt_shape['surface_eikonal_term'].squeeze(
                )) * self.opt.surf_normal_lambda
            shape_loss = shape_loss + shape_loss_dict['surface_norm_rec_loss']

        if supervise_eikonal and self.opt.eikonal_lambda > 0:
            shape_loss_dict['eikonal_term'] = eikonal_loss(
                pred_shape['eikonal_term'])[0] * self.opt.eikonal_lambda
            shape_loss = shape_loss + shape_loss_dict['eikonal_term']

        # if supervise_chamfer_dist and self.opt.chamfer_distance_lambda > 0:
        #     # st() # check pred_dict
        #     # reshape the points to b x len x dim, trunct the len dim to half to
        #     b, h, w, d = pred_shape['surface_xyz'].shape
        #     source_pts = pred_shape['surface_xyz'].reshape(b, h * w, d)
        #     b, h, w, d = gt_shape['surface_xyz'].shape
        #     target_pts = gt_shape['surface_xyz'].reshape(b, h * w, d)
        #     shape_loss_dict['chamfer_distance'] = self.chamfer_distance(
        #         source_pts, target_pts)[2] * self.opt.chamfer_distance_lambda
        #     shape_loss = shape_loss + shape_loss_dict['chamfer_distance']

        return shape_loss, shape_loss_dict

    # def chamfer_distance(self, source_cloud, target_cloud):
    #     chamferDist = ChamferDistance()

    #     dist_forward = chamferDist(source_cloud, target_cloud)
    #     dist_backward = chamferDist(source_cloud, target_cloud, reverse=True)
    #     dist_bidirectional = chamferDist(source_cloud,
    #                                      target_cloud,
    #                                      bidirectional=True)
    #     # usually use the 'dist_bidirectional' as loss
    #     return dist_forward, dist_backward, dist_bidirectional

    def calc_2d_rec_loss(self,
                         rgb_images,
                         rgb_gt,
                         high_res_rgb_gt,
                         opt,
                         loss_dict=False,
                         mode='train'):
        opt = self.opt
        cur_encoder_dict = {}

        # assert rgb_images.requires_grad or mode != 'train', 'no grad for input images'

        rec_loss = self.criterionImg(rgb_images, rgb_gt)
        lpips_loss = self.criterionLPIPS(rgb_images, rgb_gt)
        loss_psnr = self.psnr_loss((rgb_images / 2 + 0.5), (rgb_gt / 2 + 0.5),
                                   1.0)
        loss_id = torch.tensor(0., device=rgb_images.device)

        # ipdb.set_trace()
        # if discriminator is not None:  # * todo, gan loss
        #     fake_pred = discriminator(original_imgs)[0]
        #     loss_disc = F.softplus(
        #         -fake_pred).mean() * opt.w_discriminator_lambda
        # else:
        # loss_disc = torch.tensor(0) # todo

        if opt.id_lambda > 0:
            if rgb_images.shape[-1] != 256:
                # arcface_gt = self.id_loss_pool(rgb_gt)
                arcface_input = self.id_loss_pool(rgb_images)
                id_loss_gt = self.id_loss_pool(rgb_gt)
            else:
                arcface_input = rgb_images
                id_loss_gt = rgb_gt

            loss_id, _, _ = self.criterionID(arcface_input, id_loss_gt,
                                             id_loss_gt)

        loss = rec_loss * opt.l2_lambda + lpips_loss * opt.vgg_lambda + loss_id * opt.id_lambda

        # loss_ssim
        loss_ssim = kornia.losses.ssim_loss(rgb_images, rgb_gt, 5)  #?

        if loss_dict:
            cur_encoder_dict['loss_l2'] = rec_loss
            cur_encoder_dict['loss_id'] = loss_id
            cur_encoder_dict['loss_lpips'] = lpips_loss
            cur_encoder_dict['loss'] = loss
            # metrics to report, not involved in training
            cur_encoder_dict['mae'] = F.l1_loss(rgb_images, rgb_gt)
            cur_encoder_dict['PSNR'] = loss_psnr
            cur_encoder_dict['SSIM'] = 1 - loss_ssim  # Todo
            cur_encoder_dict['ID_SIM'] = 1 - loss_id

            return loss, cur_encoder_dict

        return loss, rec_loss, lpips_loss, loss_id, 0, 0

    def forward(self, *args, **kwargs):

        return self.calc_2d_rec_loss(*args, **kwargs)
