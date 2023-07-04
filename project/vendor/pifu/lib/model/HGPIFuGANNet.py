import torch
import torch.nn as nn
from ipdb import set_trace as st
import torch.autograd as autograd

from .HGPIFuNet import HGPIFuNet
from project.models.stylesdf_model import EqualLinear


def constant_init(module, val, bias=0):
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


# * for 3D GAN-based pixel-aligned & global-prior driven training
class HGPIFuNetGAN(HGPIFuNet):

    def __init__(self,
                 opt,
                 opt_stylesdf,
                 projection_mode='orthogonal',
                 error_term=nn.L1Loss()):
        super().__init__(opt, projection_mode, error_term)
        self.minT = -0.5
        self.maxT = 0.15
        self.enforce_minmax = opt.enforce_minmax  # todo
        # self.sign_error_term = nn.BCEWithLogitsLoss()

        if opt.uniform_pts_loss == 'mse':
            self.sign_error_term = nn.MSELoss()
        elif opt.uniform_pts_loss == 'bce':
            self.sign_error_term = nn.BCEWithLogitsLoss()
        elif opt.uniform_pts_loss == 'l1':
            self.sign_error_term = nn.L1Loss()

        # if opt.fix_beta:  # not learnable
        #     self.register_buffer('sigmoid_beta', torch.ones(1) * 10)
        # else:
        #     self.sigmoid_beta = nn.Parameter(
        #         10 * torch.ones(1))  # smaller -> shaper sdf
        # ? ground truth sdf is hard to learn. initialization or using other metrics?
        print('------enable clamp: ',
              self.enforce_minmax,
              '-------',
              flush=True)
        self.build_modulation_net(opt_stylesdf)
        # ref_feats = self.downsample_channel_conv(ref_feats)  # 512 -> 64
        # depth_feats = self.depth_conv(depth)

        # self.downsample_channel_conv = conv(
        #     512, 64, 3, 1)  # channel downsample img feature maps

        # in_dim = 64
        # depth_dim = 32
        # norm_layer = lambda dim: nn.InstanceNorm2d(
        #     dim, track_running_stats=False, affine=True)

        # self.depth_conv = nn.Sequential(
        #     conv3x3(1, depth_dim),
        #     ResidualBlock(depth_dim, depth_dim, norm_layer=norm_layer),
        #     conv1x1(depth_dim, depth_dim),
        # )
    def build_modulation_net(self, opt_stylesdf):

        if opt_stylesdf.L_pred_geo_modulations:
            self.local_feat_to_geo_modulations_linear = EqualLinear(
                256, 256 * 2)  # with depth
            constant_init(self.local_feat_to_geo_modulations_linear,
                          val=0,
                          bias=0)

        if opt_stylesdf.L_pred_tex_modulations:
            self.local_feat_to_tex_modulations_linear = EqualLinear(
                256, 256 * 2)  # with depth
            # self.local_feat_to_tex_modulations_linear = nn.Sequential(
            #     nn.LeakyReLU(),
            #     EqualLinear(256, 256*2)
            # )
            constant_init(self.local_feat_to_tex_modulations_linear,
                          val=0,
                          bias=0)

    def query(
            self,
            points,
            calibs,
            feat_key,
            return_eikonal=False,
            transforms=None,
            labels=None,
            #   idx=-1,
            return_feat_only=False,
            im_feat=None,
            return_projection_only=False):
        '''
        Given 3D points, query the network predictions for each point.
        Image features should be pre-computed before this call.
        store all intermediate features.
        query() function may behave differently during training/testing.
        :param points: [B, 3, N] world space coordinates of points
        :param calibs: [B, 3, 4] calibration matrices for each image
        :param transforms: Optional [B, 2, 3] image space coordinate transforms
        :param labels: Optional [B, Res, N] gt labeling
        :return: [B, Res, N] predictions for each point
        '''
        # if labels is not None:
        #     self.labels = labels

        # todo, fix z condition

        xyz = self.projection(points, calibs, transforms)
        # flip y to [-1,1] to meet grid_sample tradition. left-top pixel is x=-1, y=-1
        xyz[:, 1, :] = -1 * xyz[:, 1, :]
        # xy_debug = xyz.reshape(-1, 3, 64, 64, 24)

        xy = xyz[:, :2, :]
        z = xyz[:, 2:3, :]

        in_img = (xy[:, 0] >= -1.0) & (xy[:, 0] <= 1.0) & (
            xy[:, 1] >= -1.0) & (xy[:, 1] <= 1.0)

        z_feat = self.normalizer(z, calibs=calibs)

        if self.opt.skip_hourglass:
            tmpx_local_feature = self.index(self.tmpx, xy)

        self.intermediate_preds_list = []
        # self.intermediate_normal_list = []

        if return_eikonal:
            points.requires_grad_(True)

        # * validated, index function right

        ret_dict = {
            'proj_xy': xy,
            'depth': z,
            'in_img': in_img,
        }
        if return_projection_only:
            return ret_dict

        if im_feat is not None:
            quried_feats = self.index(im_feat, xy)
            return {
                'interp_feats': quried_feats,
                'feats': quried_feats,
                **ret_dict
            }

        for im_feat in self.im_feat_dict[feat_key]:
            # [B, Feat_i + z, N]
            # self.index
            point_local_feat_list = [self.index(im_feat, xy), z_feat]

            if return_feat_only:
                # tex = self.tex_classifier(torch.cat(point_local_feat_list, -1))
                point_local_feat = torch.cat(point_local_feat_list, 1)

                # out of image plane is always set to 0
                # pred = in_img[:, None].float() * self.surface_classifier(
                #     point_local_feat)  # * pred: sdf
                return {
                    'feats': point_local_feat_list[0],
                    'z_condition': point_local_feat_list[1],
                    'point_local_feat': point_local_feat,
                    # 'in_img': in_img,
                    # 'proj_xy': xy,
                    # 'depth': z,
                    **ret_dict
                    # 'pred_sdf': pred
                }
                # return [point_local_feat_list, None]

            if self.opt.skip_hourglass:
                point_local_feat_list.append(tmpx_local_feature)

            # st()
            point_local_feat = torch.cat(point_local_feat_list, 1)

            # out of image plane is always set to 0
            pred = in_img[:, None].float() * self.surface_classifier(
                point_local_feat)  # * pred: sdf

            # if self.opt.debug:
            #     feat_mean = point_local_feat[0].mean()

            if self.opt.enforce_minmax and not self.opt.debug:
                # todo, necessary?
                # if self.opt.debug:
                # with torch.no_grad():
                #     loss = self.error_term(labels, pred)
                labels = torch.clamp(labels, self.minT, self.maxT)
                pred = torch.clamp(pred, self.minT, self.maxT)

            if return_eikonal:  # todo? breaks original logics
                eikonal_term = autograd.grad(
                    outputs=pred,
                    inputs=points,
                    grad_outputs=torch.ones_like(pred),
                    create_graph=True)[0]
            else:
                eikonal_term = None

            self.intermediate_preds_list.append([pred, eikonal_term])
            # else:
            #     self.intermediate_preds_list.append(pred)

        self.preds = self.intermediate_preds_list[-1]

    def validate_points_shape(self):
        for preds, eikonal_terms in self.intermediate_preds_list:
            print(preds.shape)

    def get_error(self):
        """Hourglass has its own intermediate supervision scheme
        preds shape: B k N, k in [1,2]

        Returns:
            dict: dict of all losses
        """

        # sdf, l1 loss
        error = 0
        eikonal_error = 0
        for preds, eikonal_terms in self.intermediate_preds_list:

            surface_points_num = self.points['surface_pts'].shape[-1]
            all_pts_num = preds.shape[
                -1]  # preds return B,1,N sdf/occupancy shape

            pred_surface_points_sdf, pred_uniform_points_sdf = torch.split(
                preds, [surface_points_num, all_pts_num - surface_points_num],
                -1)

            if eikonal_terms is not None:
                surface_points_norm, uniform_points_norm = torch.split(
                    eikonal_terms,
                    [surface_points_num, all_pts_num - surface_points_num], -1)
                surface_norm_labels = self.labels['surface_norm_labels']
                surf_norm_rec = self.error_term(surface_points_norm,
                                                surface_norm_labels)
            else:
                surf_norm_rec = torch.tensor(0, device=preds.device)

            points_in_out_labels = self.labels[
                'uniform_pts_sdf_labels']  # 0 or 1

            # surface points supervision
            # surf_rec = self.error_term(surface_points_sdf, torch.zeros_like(surface_points_sdf))
            surf_rec = torch.tensor(0, device=preds.device)

            # uniform points supervision

            # * apply mask to prediction
            # for v in [surface_eikonal_term]:
            #     if v.ndim == 5:
            #         mask = fg_mask.unsqueeze(
            #             -2)
            #     else:
            #         mask = fg_mask # b 64 64 1
            #     v = v * mask.expand_as(v)  # 1 for fg, 0 for bg

            # st()
            for v in [points_in_out_labels, pred_uniform_points_sdf]:
                v = v * self.valid_mask['uniform_pts_valid_mask']

            # st()
            # uniform_rec = self.sign_error_term(self.sigmoid_beta*uniform_points_sdf, points_in_out_labels)
            uniform_rec = self.sign_error_term(pred_uniform_points_sdf,
                                               points_in_out_labels)
            # uniform_rec = torch.nn.functional.l1_loss(uniform_points_sdf, points_in_out_labels)
            if self.return_eikonal:
                uniform_eikonal = ((uniform_points_norm.norm(dim=1) -
                                    1)**2).mean()
            else:
                uniform_eikonal = torch.tensor(
                    0, device=points_in_out_labels.device)

            error += surf_rec * self.opt.lambda_g1 + surf_norm_rec * self.opt.lambda_g2 + uniform_rec * self.opt.lambda_l + uniform_eikonal * self.opt.lambda_e

        error /= len(self.intermediate_preds_list)
        # eikonal_error /= len(self.intermediate_preds_list)

        total_error = error + eikonal_error

        # todo
        # surface norm rec
        # normal_loss = self.error_term()

        # todo
        # color error, also l1

        # return error, {
        #     'total_loss': error.item(),
        #     'g1': surf_rec.item(),
        #     'g2': surf_norm_rec.item(),
        #     'l': uniform_rec.item(),
        #     'e': uniform_eikonal.item()
        # }
        return error, {
            'total_loss': error,
            'g1': surf_rec,
            'g2': surf_norm_rec,
            'l': uniform_rec,
            'e': uniform_eikonal
        }

    def preprocess_points(self, data_batch, flatten=True, fg_mask=False):
        # flatten & mask
        """perprocess points shape for later use
        :output points shape: [B, 3, N] world space coordinates of points

        Args:
            data_batch (_type_): batch from data loader
        """
        # todo, move mask to loss
        if self.add_fg_mask:
            fg_indices = mask.nonzero()
            st()
            xyz = xyz[fg_indices]
            surface_eikonal_term = surface_eikonal_term[fg_indices]

            sdf = sdf[fg_indices]
            normalized_pts = normalized_pts[fg_indices]

        B = data_batch['points'].shape[0]
        # 2. flatten and reshape
        if flatten:  # B,h,w,steps,3 -> B, 3, -1; for interpolation?
            data_batch['points'] = data_batch['points'].reshape(B, -1,
                                                                3).permute(
                                                                    0, 2, 1)
            data_batch['xyz'] = data_batch['xyz'].reshape(B, -1,
                                                          3).permute(0, 2, 1)

            data_batch['surface_eikonal_term'] = data_batch[
                'surface_eikonal_term'].reshape(B, -1, 3).permute(0, 2, 1)
            data_batch['sdf'] = data_batch['sdf'].reshape(B, -1,
                                                          3).permute(0, 2, 1)

    # * todo, add GAN features
    def forward(
            self,
            data_batch,
            idx,
            transforms=None,
            # labels=None,
            gan_features=None):

        # self.preprocess_points(data_batch)

        # retrieve input points
        uniform_points, images, calibs, surface_points, fg_mask = [
            data_batch[k]
            for k in ['uniform_pts', 'gen_imgs', 'calibs', 'xyz', 'mask']
        ]

        # retrieve gt
        uniform_sdf_labels, uniform_points_valid_mask, surface_eikonal_term = [
            data_batch[k] for k in [
                'uniform_points_sdf', 'uniform_points_valid_mask',
                'surface_eikonal_term'
            ]
        ]

        # * apply fg mask

        # todo, add fgmask
        # for v in [surface_points, surface_eikonal_term]:

        # st()
        # * reshape points to B,3,N
        B = surface_points.shape[0]
        surface_points = surface_points.reshape(B, -1, 3).permute(0, 2,
                                                                  1)  # B,3,N
        uniform_points = uniform_points.reshape(B, -1, 3).permute(0, 2, 1)
        uniform_sdf_labels = uniform_sdf_labels.reshape(B, -1,
                                                        1).permute(0, 2, 1)
        uniform_sdf_labels = torch.gt(uniform_sdf_labels,
                                      0).type(torch.float)  # todo
        # sdf_labels = sdf

        # surface_points_num = surface_points.shape[-1]

        # rand_perm_index = torch.randperm(uniform_points.shape[-1], device=surface_points.device)
        # uniform_points = uniform_points[..., rand_perm_index[:4096]]
        # uniform_pts_sdf_labels = sdf_labels[..., rand_perm_index[:4096]]

        if self.return_eikonal:
            surface_points.requires_grad_(True)
            uniform_points.requires_grad_(True)
            # images.requires_grad_(True)

        self.points = {
            'surface_pts': surface_points,
            'uniform_pts': uniform_points  # todo, remove with near_surf_points
        }

        self.labels = {
            'surface_norm_labels': surface_eikonal_term,  # todo, reshape
            'uniform_pts_sdf_labels': uniform_sdf_labels
        }
        self.valid_mask = {
            'surface_pts_valid_mask': data_batch['mask'],
            'uniform_pts_valid_mask': data_batch['uniform_points_valid_mask']
        }  # todo, merge into G

        # *debug
        # points = surface_points # todo
        # points = uniform_points
        points = torch.cat(
            [self.points['surface_pts'], self.points['uniform_pts']],
            -1)  # B, 3, N, N=4096*2 for now

        self.filter(images)
        # Phase 2: point query
        # points = self.preprocess_points(points) # 1. normalize; 2. flip y to [-1,1] to meet grid_sample order
        self.query(
            points=points,  # todo
            calibs=calibs,
            transforms=transforms,
            labels=None,
            return_eikonal=False,
            idx=idx)  # todo, flags

        # todo Phase 3: combine with global prior

        # get the prediction
        res = self.get_preds()

        # get the error
        error, error_info = self.get_error()  # dict

        return res, error, error_info
