import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np
from pdb import set_trace as st
from skimage.measure import marching_cubes
import trimesh

from project.utils.camera_utils import make_homo_cam_matrices, make_homo_pts

from . import align_volume

sys.path.append('project/vendor/pifu')

from lib.mesh_util import *
from lib.sample_util import *
from lib.train_util import *
from lib.data import *


class UniformBoxWarp(nn.Module):

    def __init__(self, sidelength):
        super().__init__()
        self.scale_factor = 2 / sidelength

    def forward(self, coordinates):
        return coordinates * self.scale_factor


# from mmcv
def constant_init(module, val, bias=0):
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


# Basic SIREN fully connected layer
class LinearLayer(nn.Module):

    def __init__(self,
                 in_dim,
                 out_dim,
                 bias=True,
                 bias_init=0,
                 std_init=1,
                 freq_init=False,
                 is_first=False):
        super().__init__()
        if is_first:
            self.weight = nn.Parameter(
                torch.empty(out_dim, in_dim).uniform_(-1 / in_dim, 1 / in_dim))
        elif freq_init:
            self.weight = nn.Parameter(
                torch.empty(out_dim,
                            in_dim).uniform_(-np.sqrt(6 / in_dim) / 25,
                                             np.sqrt(6 / in_dim) / 25))
        else:
            self.weight = nn.Parameter(
                0.25 * nn.init.kaiming_normal_(torch.randn(out_dim, in_dim),
                                               a=0.2,
                                               mode='fan_in',
                                               nonlinearity='leaky_relu'))

        self.bias = nn.Parameter(
            nn.init.uniform_(torch.empty(out_dim),
                             a=-np.sqrt(1 / in_dim),
                             b=np.sqrt(1 / in_dim)))

        self.bias_init = bias_init
        self.std_init = std_init

    def forward(self, input):
        out = self.std_init * F.linear(input, self.weight,
                                       bias=self.bias) + self.bias_init

        return out


# Siren layer with frequency modulation and offset
class FiLMSiren(nn.Module):

    def __init__(self, in_channel, out_channel, style_dim, is_first=False):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel

        if is_first:
            self.weight = nn.Parameter(
                torch.empty(out_channel, in_channel).uniform_(-1 / 3, 1 / 3))
        else:
            self.weight = nn.Parameter(
                torch.empty(out_channel,
                            in_channel).uniform_(-np.sqrt(6 / in_channel) / 25,
                                                 np.sqrt(6 / in_channel) / 25))

        self.bias = nn.Parameter(
            nn.Parameter(
                nn.init.uniform_(torch.empty(out_channel),
                                 a=-np.sqrt(1 / in_channel),
                                 b=np.sqrt(1 / in_channel))))
        self.activation = torch.sin

        self.gamma = LinearLayer(style_dim,
                                 out_channel,
                                 bias_init=30,
                                 std_init=15)
        self.beta = LinearLayer(style_dim,
                                out_channel,
                                bias_init=0,
                                std_init=0.25)

    def forward(self, input, style):
        batch, features = style.shape
        out = F.linear(input, self.weight, bias=self.bias) # out: B H W num_steps C
        gamma = self.gamma(style).reshape(batch, 1, 1, 1, features)
        beta = self.beta(style).reshape(batch, 1, 1, 1, features)
        
        if out.shape[2] == 128: # optimize memory use to support batch inference with 15GB memory (Colab T4 GPU)
            out_holder = torch.zeros_like(out)
            sub_batch = 16 # tune it here 
            assert out.shape[2] % sub_batch == 0
            for sub_batch_idx in range(0, out.shape[-2], sub_batch):
                out_holder[..., sub_batch_idx:sub_batch_idx+sub_batch, :] = self.activation(gamma * out[..., sub_batch_idx:sub_batch_idx+sub_batch, :] + beta)
            out = out_holder 
        else:
            out = self.activation(gamma * out + beta)

        return out


# Siren Generator Model
class SirenGenerator(nn.Module):

    def __init__(self,
                 opt,
                 D=8,
                 W=256,
                 style_dim=256,
                 input_ch=3,
                 input_ch_views=3,
                 output_ch=4,
                 output_features=True,
                 scene_scale=0.12,
                 **kwargs):
        super(SirenGenerator, self).__init__()
        self.opt = opt
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.style_dim = style_dim
        self.output_features = output_features

        self.pts_linears = nn.ModuleList(
            [FiLMSiren(3, W, style_dim=style_dim, is_first=True)] + \
            [FiLMSiren(W, W, style_dim=style_dim) for i in range(D-1)])

        self.views_linears = FiLMSiren(input_ch_views + W,
                                       W,
                                       style_dim=style_dim)
        self.rgb_linear = LinearLayer(W, 3, freq_init=True)
        self.sigma_linear = LinearLayer(W, 1, freq_init=True)

    def forward_generator(self, input_pts, styles, conditions: dict = None):
        # returns global features
        mlp_out = input_pts.contiguous()

        return_feats = self.opt.return_feats
        if return_feats:
            feats_out = []

        if styles.ndim == 3:  # w/w+ space
            for i in range(len(self.pts_linears)):  # B,9,256
                mlp_out = self.pts_linears[i](mlp_out, styles[:, i])
                if return_feats and (i + 1) in self.opt.return_feats_layers:
                    feats_out.append(mlp_out.detach())

                # try the texture first
                # Sept 28, move to the forward_tex(), since this will affect the geo output
                # if self.opt.local_modulation_layer_in_backbone and i == self.opt.local_modulation_layer and bool(
                #         conditions):  # condition none empty
                #     alpha, beta = conditions['tex']
                #     mlp_out = (alpha + 1) * mlp_out + beta

        else:  # z space, 1 * self.style_dim
            for i in range(len(self.pts_linears)):
                mlp_out = self.pts_linears[i](mlp_out, styles)
        if return_feats:
            return mlp_out, feats_out
        return mlp_out

    def forward_backbone(self, input_pts, styles, local_feats=None):
        """extracts information from given points and styles, and support basic residual fusion
        """
        mlp_out = self.forward_generator(input_pts, styles)

        # if local_feats is not None:
        #     mlp_out = mlp_out + local_feats

        return mlp_out

    def forward_geo(self, mlp_out):
        sdf = self.sigma_linear(mlp_out)
        return sdf

    def forward_tex(self, mlp_out, input_views, styles, conditions=None):

        if isinstance(mlp_out, dict):  # add local condition
            mlp_out, conditions = mlp_out['mlp_out'], mlp_out.get(
                'conditions', None)

        # move before view_linears to help convergence and avoid overfitting, that mlp helps with regularization.
        if not self.opt.local_modulation_layer_in_backbone and self.opt.local_modulation_layer and bool(
                conditions):  # condition none empty
            alpha, beta = conditions['tex']
            mlp_out = (alpha + 1) * mlp_out + beta

        mlp_out = torch.cat([mlp_out, input_views], -1)

        if styles.shape[-1] != self.style_dim:  # 2048
            view_dir_styles = styles[..., -self.style_dim:]  # todo
        elif styles.ndim == 3:
            view_dir_styles = styles[:, -1]
        else:
            view_dir_styles = styles

        out_features = self.views_linears(
            mlp_out,
            view_dir_styles)  # todo, how to deal here; init with mean(w) now

        rgb = self.rgb_linear(
            out_features)  # contains all view-dependent informaiton

        return rgb, out_features

    def forward(self, net_inputs, styles, residuals=None):
        # === extract G features ====
        if not net_inputs.requires_grad:
            net_inputs.requires_grad_(True)  # for inversion tasks only.

        input_pts, input_views = torch.split(
            net_inputs, [self.input_ch, self.input_ch_views], dim=-1)
        mlp_out = self.forward_backbone(input_pts, styles)

        if self.opt.return_feats:
            mlp_out, all_feats = mlp_out

        # === reconstruct geometry ====
        sdf = self.forward_geo(mlp_out)

        # === reconstruct color ====
        rgb, out_features = self.forward_tex(mlp_out, input_views, styles)

        # === return
        outputs = torch.cat([rgb, sdf], -1)
        if self.output_features:
            outputs = torch.cat([outputs, out_features], -1)
        if self.opt.return_feats:
            return outputs, all_feats
        return outputs


class SirenLocalGlobal(nn.Module):
    # a higher level class with both L and G network
    def __init__(self,
                 D=8,
                 W=256,
                 style_dim=256,
                 input_ch=3,
                 input_ch_views=3,
                 output_ch=4,
                 output_features=True,
                 scene_scale=0.12,
                 local_options=None,
                 opt=None):

        super().__init__()
        # add local branch to the rendering process

        # todo, should move the attributes out?
        self.opt = opt  # rendering option
        self.netGlobal = SirenGenerator(opt, D, W, style_dim, input_ch,
                                        input_ch_views, output_ch,
                                        output_features, scene_scale)

        # pifu modules, lazy import 
        from lib.model import HGPIFuNetGAN, HGPIFuNetGANResidual, HGPIFuNetGANResidualResnetFC

        if opt.netLocal_type == 'HGPIFuNetGAN':
            self.netLocal = HGPIFuNetGAN(local_options, opt, 'projection')
        elif opt.netLocal_type == 'HGPIFuNetGANResidual':
            self.netLocal = HGPIFuNetGANResidual(local_options, opt,
                                                 'projection')
        elif opt.netLocal_type == 'HGPIFuNetGANResidualResnetFC':
            self.netLocal = HGPIFuNetGANResidualResnetFC(
                local_options, opt, 'projection')
        else:
            raise NotImplementedError('netLocal_type')

        # local_feat_to_tex_modulations = None
        # pass
        # assert not (
        #     opt.use_G_geo and opt.use_L_geo
        # ), 'only enable one geo prediction branch for ablation study'

        # if opt.use_L_geo_as_residual:
        #     assert opt.use_L_geo, 'L geo must be enabled'

    def forward_backbone(self, input_pts, styles, local_data_batch):
        """extracts output from two subnetworks, fuse and return organized forward output.
        
        return: dictionary of {
            global_output: 
            local_output: 
        }
        """

        local_modulation_conditions = dict()
        if local_data_batch is not None and 'sampling' not in local_data_batch:
            local_output = self.forward_local(
                local_data_batch)  # feats, z_condition, perd_sdf

            if self.opt.L_pred_tex_modulations:
                # local_tex_modulations = self.local_feat_to_tex_modulations_linear(local_output['point_local_feat'])
                local_tex_modulations = self.netLocal.local_feat_to_tex_modulations_linear(
                    local_output['feats'])
                # st()
                tex_alpha, tex_beta = torch.split(local_tex_modulations,
                                                  256,
                                                  dim=-1)
                local_modulation_conditions.update(tex=[tex_alpha, tex_beta])
                local_output['tex_modulate_conditions'] = True

            if self.opt.L_pred_geo_modulations:
                local_geo_modulations = self.netLocal.local_feat_to_geo_modulations_linear(
                    local_output['feats'])  # remove depth input for now
                geo_alpha, geo_beta = torch.split(local_geo_modulations,
                                                  256,
                                                  dim=-1)
                local_modulation_conditions.update(geo=[geo_alpha, geo_beta])
                local_output['geo_modulate_conditions'] = True

        else:
            local_output = None

        # global_feats = self.netGlobal.forward_generator(
        #     input_pts, styles)
        global_feats = self.netGlobal.forward_generator(
            input_pts, styles, local_modulation_conditions) # local_modulation_conditions not used here

        if self.opt.return_feats:
            assert isinstance(global_feats, tuple)
            global_output = {
                'feats': global_feats[0],
                'all_feats': global_feats[1]
            }
        else:
            global_output = {'feats': global_feats}

        ret_dict = dict(global_output=global_output, local_output=local_output)
        if local_modulation_conditions is not None:
            ret_dict.update(
                local_modulation_conditions=local_modulation_conditions)

        return ret_dict

    def retrieve_feats_for_rendering(self, forward_out: dict,
                                     sample_mode: bool):
        """distribute feats according to strategy defined in option

        Args:
            forward_out (dict): returned from forward_backbone()
        """
        geo_predictition_strategy = self.opt.geo_predictition_strategy
        tex_predictition_strategy = self.opt.tex_predictition_strategy

        global_feats = forward_out['global_output']['feats']

        # if sample_mode:
        #     return dict(feats_to_geo=global_feats, feats_to_tex=global_feats)

        if forward_out['local_output'] is None or sample_mode:
            return dict(feats_to_geo=global_feats, feats_to_tex=global_feats)
        else:
            local_feats = forward_out['local_output']['feats']

        # * return feature for training & inference
        # geometry feats
        if 'global' in geo_predictition_strategy:
            feats_to_geo = global_feats
            if 'local' in geo_predictition_strategy:
                assert 'local_modulation_conditions' in forward_out
                alpha, beta = forward_out['local_modulation_conditions']['geo']
                feats_to_geo = (alpha + 1) * feats_to_geo + beta
        else:
            raise NotImplementedError('not adopted in this paper')
            assert 'local' in geo_predictition_strategy
            feats_to_geo = local_feats

        # ! todo, deprecated.
        # texture feats
        if 'global' in tex_predictition_strategy:
            feats_to_tex = global_feats
            if 'local' in tex_predictition_strategy and not self.opt.L_pred_tex_modulations:
                raise DeprecationWarning('this feature is deprecated for now?')
                feats_to_tex = self.feat_fusion(feats_to_tex,
                                                local_feats)  # directly add
        else:
            assert 'local' in tex_predictition_strategy
            feats_to_tex = local_feats

        # ! retrieve local features for fusion
        feats_to_tex = dict(mlp_out=feats_to_tex)
        if 'local_modulation_conditions' in forward_out and 'tex' in forward_out[
                'local_modulation_conditions']:
            feats_to_tex.update(
                dict(conditions=forward_out['local_modulation_conditions']))

        ret_dict = dict(feats_to_geo=feats_to_geo, feats_to_tex=feats_to_tex)

        # if 'all_feats' in forward_out['global_output']:
        #     ret_dict.update('all_feats': forward_out['global_output'])

        return ret_dict

    def feat_fusion(self,
                    source_feats,
                    target_feats,
                    fusion_strategy='residual'):
        if fusion_strategy == 'residual':
            return source_feats + target_feats

        raise NotImplementedError('implement more fusion methods')

    def forward_local(self, data_batch):
        """a lint version of hgpifu_net which only do the feature extraction stuff
        
        return: feats of local points
        """
        if 'feats' in data_batch and data_batch[
                'feats'] is not None:  # already computed local feats externally
            return data_batch

        # 2D alignment behaviours below. for ablations.
        points, images, calibs = [
            data_batch[k] for k in ['world_space_pts', 'gen_imgs', 'calibs']
        ]

        reshape_flag = False
        if points.ndim == 5:
            B, H, W, S, _ = points.shape
            reshape_flag = True

        points = points.reshape(B, -1, 3).permute(0, 2, 1)

        self.netLocal.filter(images)
        local_output = self.netLocal.query(
            points=points,  # todo
            calibs=calibs,
            return_eikonal=False,
            return_feat_only=True)  # B, N, 256
        # local_feats, z_condition, pred_sdf = (
        #     local_output[k]
        #     for k in ['feats', 'z_condition', 'pred_sdf'])

        if reshape_flag:
            for k, v in local_output.items():
                local_output[k] = v.reshape(B, -1, H, W,
                                            S).permute(0, 2, 3, 4, 1)
            # local_output['feats'] = local_output['feats'].reshape( B, -1, H, W, S).permute(0, 2, 3, 4, 1)
            # local_output['pred_sdf'] = local_output['pred_sdf'].reshape(
            #     B, -1, H, W, S).permute(0, 2, 3, 4, 1)

        return local_output  # todo, make dict

    # todo, change params
    def forward_rendering(self, feats_for_render_dict, input_views, styles):
        """return texture and geometry from given input features

        Args:
            feats_to_geo (torch.Tensor): feats to geometry mlp
            feats_to_tex (torch.Tensor): feats to texture mlp

        Returns:
            tuple: geometry, texture
        """
        # todo, now just use global branch rendering mlp

        # * todo
        feats_to_geo, feats_to_tex = [
            feats_for_render_dict[k] for k in ['feats_to_geo', 'feats_to_tex']
        ]

        # === reconstruct color ====
        rgb, out_features = self.feats_to_tex_query(feats_to_tex, input_views,
                                                    styles)

        # === reconstruct geometry ====
        # if not self.opt.use_L_geo:
        # * ignore L geometry for now
        sdf = self.feats_to_geo_query(feats_to_geo)
        # else:
        #     sdf = forward_backbone_output['local_output']['pred_sdf']
        #     if self.opt.use_L_geo_as_residual:
        #         sdf = self.forward_geo(global_feats) + sdf
        #         # todo, use norm?
        # else:
        #     sdf = self.feats_to_tex_query(fused_feats)

        # === return
        outputs = torch.cat([rgb, sdf], -1)
        if self.netGlobal.output_features:
            outputs = torch.cat([outputs, out_features], -1)
        return outputs

    # * use renderer MLP for rendering query
    def feats_to_geo_query(self, *args):
        return self.netGlobal.forward_geo(*args)

    def feats_to_tex_query(self, *args, **kwargs):
        return self.netGlobal.forward_tex(*args, **kwargs)

    def forward(self,
                net_inputs,
                styles,
                local_data_batch: dict = None,
                sample_mode=False
                ):  # todo how to send params to deep sub-class instance?
        # inference_flag used to 1. sample  2. do inference
        # todo, support L geometry inference.

        # === extract inputs ====
        input_pts, input_views = torch.split(
            net_inputs,
            [self.netGlobal.input_ch, self.netGlobal.input_ch_views],
            dim=-1)
        # B = input_pts.shape[0]

        # === extract output of G and L output ===
        forward_backbone_output = self.forward_backbone(
            input_pts, styles, local_data_batch)

        # === extract feats for geo and tex (under current fusion strategy)
        feats_for_render_dict = self.retrieve_feats_for_rendering(
            forward_backbone_output, sample_mode)

        # === make predictions ===
        rendering_outputs = self.forward_rendering(feats_for_render_dict,
                                                   input_views, styles)
        if self.opt.return_feats:
            return dict(raw=rendering_outputs,
                        all_feats=forward_backbone_output['global_output']
                        ['all_feats'])
        return rendering_outputs


class SirenGeneratorDDF(SirenGenerator):

    def __init__(self,
                 D=8,
                 W=256,
                 style_dim=256,
                 input_ch=3,
                 input_ch_views=3,
                 output_ch=4,
                 output_features=True):
        super().__init__(D=8,
                         W=256,
                         style_dim=256,
                         input_ch=3,
                         input_ch_views=3,
                         output_ch=4,
                         output_features=True)

    def forward(self, x, styles, **kwargs):
        ret_dict = {
            'feats': torch.zeros(x.shape[0], x.shape[1], 0, device=x.device)
        }
        input_pts, input_views = torch.split(
            x, [self.input_ch, self.input_ch_views], dim=-1)
        mlp_out = input_pts.contiguous()

        if styles.ndim == 3:
            for i in range(len(self.pts_linears)):  # B,9,256
                mlp_out = self.pts_linears[i](mlp_out, styles[:, i])

                if kwargs.get('multi_layer_loss', False):
                    if kwargs.get('feat_layer', -1) <= i:  # fet last N layers
                        ret_dict['feats'] = torch.cat(
                            [ret_dict['feats'], mlp_out], -1)
                else:
                    if kwargs.get('feat_layer', -1) == i:
                        ret_dict['feats'] = torch.cat(
                            [ret_dict['feats'], mlp_out], -1)

        else:  # 1 * self.style_dim, original setting
            for i in range(len(self.pts_linears)):
                mlp_out = self.pts_linears[i](mlp_out, styles)

        sdf = self.sigma_linear(mlp_out)
        # st()  # compare out[1] and thumb_img shape: 64 and 128
        mlp_out = torch.cat([mlp_out, input_views], -1)
        if styles.shape[-1] != self.style_dim:  # 2048
            view_dir_styles = styles[..., -self.style_dim:]  # todo
        elif styles.ndim == 3:
            view_dir_styles = styles[:, -1]
        else:
            view_dir_styles = styles
        out_features = self.views_linears(
            mlp_out,
            view_dir_styles)  # todo, how to deal here; init with mean(w) now
        rgb = self.rgb_linear(out_features)
        outputs = torch.cat([rgb, sdf], -1)

        if kwargs.get('return_x', False):
            if kwargs.get('feat_layer',
                          -1) == -1 or kwargs['multi_layer_loss']:
                # ret_dict['feat'] = rgb_feat
                ret_dict['feats'] = torch.cat(
                    [ret_dict['feats'], out_features], -1)
            ret_dict.update({
                'out': outputs,
            })
            return ret_dict

        if self.output_features:
            outputs = torch.cat([outputs, out_features], -1)
        return outputs


# Full volume renderer
class VolumeFeatureRenderer(nn.Module):

    def __init__(self, opt, style_dim=256, out_im_res=64, mode='train'):
        super().__init__()
        self.test = mode != 'train'
        self.opt = opt
        self.perturb = opt.perturb
        self.offset_sampling = not opt.no_offset_sampling  # Stratified sampling used otherwise
        self.N_samples = opt.N_samples
        self.raw_noise_std = opt.raw_noise_std
        self.return_xyz = opt.return_xyz
        # self.return_sdf = opt.return_sdf
        self.return_sdf = True
        self.static_viewdirs = opt.static_viewdirs
        self.z_normalize = not opt.no_z_normalize
        self.out_im_res = out_im_res
        self.spatial_ss = opt.spatial_super_sampling_factor
        # self.spatial_ss = spatial_ss
        self.force_background = opt.force_background
        self.with_sdf = not opt.no_sdf
        self.add_fg_mask = opt.add_fg_mask
        if 'no_features_output' in opt.keys():
            self.output_features = False
        else:
            self.output_features = True

        if self.with_sdf:
            self.sigmoid_beta = nn.Parameter(0.1 * torch.ones(1))

        # create meshgrid to generate rays
        i, j = torch.meshgrid(
            torch.linspace(0.5, self.out_im_res - 0.5,
                           self.out_im_res * self.spatial_ss),
            torch.linspace(0.5, self.out_im_res - 0.5,
                           self.out_im_res * self.spatial_ss))

        self.register_buffer('i', i.t().unsqueeze(0),
                             persistent=False)  # 1, H, W
        self.register_buffer('j', j.t().unsqueeze(0), persistent=False)

        # auxiliary for surface sampling
        # i_shift, j_shift = torch.meshgrid(
        #     torch.linspace(0, self.out_im_res - 1, self.out_im_res),
        #     torch.linspace(0, self.out_im_res - 1, self.out_im_res))

        self.register_buffer('grid',
                             torch.stack((i, j), 2).float(),
                             persistent=False)  # W(x), H(y), 2
        self.register_buffer('normalized_grid',
                             (self.grid - self.out_im_res * .5) /
                             (self.out_im_res * .5),
                             persistent=False)  # W(x), H(y), 2

        # create integration values
        if self.offset_sampling:
            t_vals = torch.linspace(0.,
                                    1. - 1 / self.N_samples,
                                    steps=self.N_samples).reshape(1, 1, 1, -1)
        else:  # Original NeRF Stratified sampling
            t_vals = torch.linspace(0., 1.,
                                    steps=self.N_samples).reshape(1, 1, 1, -1)

        self.register_buffer('t_vals', t_vals, persistent=False)
        self.register_buffer('inf', torch.Tensor([1e10]), persistent=False)
        self.register_buffer('zero_idx',
                             torch.LongTensor([0]),
                             persistent=False)

        if self.test:
            self.perturb = False
            self.raw_noise_std = 0.

        self.channel_dim = -1
        self.samples_dim = 3
        self.input_ch = 3
        self.input_ch_views = 3
        self.feature_out_size = opt.width

        # set Siren Generator model
        # if opt.ddf:
        #     SirenModel = SirenGeneratorDDF
        # else:
        # SirenModel = SirenGenerator

        self.grid_warper = UniformBoxWarp(opt.camera.dist_radius * 2)
        self.grid_un_warper = UniformBoxWarp(1 / opt.camera.dist_radius *
                                             2)  # used in local branch

        # local model
        if opt.enable_local_model:
            feat_extraction_net = SirenLocalGlobal
            self.enable_local_model = True
        else:
            self.enable_local_model = False
            feat_extraction_net = SirenGenerator

        self.network = feat_extraction_net(
            opt=opt,
            D=opt.depth,
            W=opt.width,
            style_dim=style_dim,
            input_ch=self.input_ch,
            output_ch=4,
            input_ch_views=self.input_ch_views,
            output_features=self.output_features,
            local_options=opt.pifu if self.enable_local_model else None)
        # enable_global=not opt.disable_global_model,
        #
        self.register_buffer('B_MAX',
                             torch.Tensor([opt.camera.dist_radius] * 3),
                             persistent=False)
        self.register_buffer('B_MIN',
                             -torch.Tensor([opt.camera.dist_radius] * 3),
                             persistent=False)

    @torch.no_grad()
    def lms2rays(self, lms, focal, out_im_res=256):
        """2D lms output coordinates to right hand coord system

        Args:
            lms (torch.Tensor): 2D LMS coords
            focal (float): focal
        """
        lms_num = lms.shape[-2]  # B 68/82/7 2
        i, j = torch.split(lms, 2, dim=-1)  # B N 1
        dirs = torch.stack([
            i / focal, -j / focal,
            -torch.ones_like(i).expand(lms.shape[0], lms_num, 1)
        ], -1)

        return dirs

    @torch.no_grad()
    def get_rays(self, focal, c2w, dirs=None):
        # K.inv()
        if dirs is None:
            dirs = torch.stack(
                [(self.i - self.out_im_res * .5) / focal,
                 -(self.j - self.out_im_res * .5) / focal,
                 -torch.ones_like(self.i).expand(
                     focal.shape[0], self.out_im_res * self.spatial_ss,
                     self.out_im_res * self.spatial_ss)], -1)
            #  focal.shape[0], self.out_im_res, self.out_im_res)], -1)
        else:
            assert dirs.ndim == 4, 'check dimention'

        rays_d = torch.sum(
            dirs[..., None, :] * c2w[:, None, None, :3, :3],  #c2w:[1,3,4]
            -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
        # check rays_d.shape:[1, 128, 128, 3], type

        # Translate camera frame's origin to the world frame. It is the origin of all rays.
        rays_o = c2w[:, None, None, :3, -1].expand(rays_d.shape)
        if self.static_viewdirs:
            viewdirs = dirs
        else:
            viewdirs = rays_d

        return rays_o, rays_d, viewdirs

    def get_eikonal_term(self, pts, sdf):
        eikonal_term = autograd.grad(outputs=sdf,
                                     inputs=pts,
                                     grad_outputs=torch.ones_like(sdf),
                                     create_graph=True)[0]

        return eikonal_term

    def sdf_activation(self, input):  # why?
        sigma = torch.sigmoid(input / self.sigmoid_beta) / self.sigmoid_beta

        return sigma

    def volume_integration(
            self,
            raw,
            z_vals,
            rays_d,
            pts,
            return_eikonal,
            return_surface_eikonal,
            return_mesh=True,
            #    mesh_with_shading=True,
            c2w=None,
            no_force_stop=False,
            **kwargs):

        if isinstance(raw, dict):
            raw = raw['raw']

        dists = z_vals[..., 1:] - z_vals[..., :-1]
        rays_d_norm = torch.norm(rays_d.unsqueeze(self.samples_dim),
                                 dim=self.channel_dim)
        # dists still has 4 dimensions here instead of 5, hence, in this case samples dim is actually the channel dim
        if not no_force_stop:
            dists = torch.cat(
                [dists, self.inf.expand(rays_d_norm.shape)],
                self.channel_dim)  # [N_rays, N_samples]
        else:  # for query reference views
            dists = torch.cat([dists, dists[..., 0:1]],
                              self.channel_dim)  # [N_rays, N_samples]
        dists = dists * rays_d_norm

        # If sdf modeling is off, the sdf variable stores the
        # pre-integration raw sigma MLP outputs.
        if self.output_features:
            rgb, sdf, features = torch.split(raw,
                                             [3, 1, self.feature_out_size],
                                             dim=self.channel_dim)
        else:
            rgb, sdf = torch.split(raw, [3, 1], dim=self.channel_dim)

        noise = 0.
        if self.raw_noise_std > 0.:
            noise = torch.randn_like(sdf) * self.raw_noise_std

        if self.with_sdf:
            sigma = self.sdf_activation(-sdf)  # * translate to density

            if return_eikonal:
                eikonal_term = self.get_eikonal_term(pts, sdf)
            else:
                eikonal_term = None

            sigma = 1 - torch.exp(
                -sigma * dists.unsqueeze(self.channel_dim))  # alpha
        else:
            sigma = sdf
            eikonal_term = None

            sigma = 1 - torch.exp(
                -F.softplus(sigma + noise) * dists.unsqueeze(self.channel_dim))

        visibility = torch.cumprod(
            torch.cat([
                torch.ones_like(
                    torch.index_select(sigma, self.samples_dim,
                                       self.zero_idx)), 1. - sigma + 1e-10
            ], self.samples_dim), self.samples_dim)
        # st() # check dirs type # check visibility: are they relative to the view point
        visibility = visibility[..., :-1, :]
        weights = sigma * visibility  # hit probability

        if self.return_sdf:
            sdf_out = sdf
        else:
            sdf_out = None

        if self.force_background and not no_force_stop:  # only apply this to batch with far samples
            weights[...,
                    -1, :] = 1 - weights[..., :-1, :].sum(self.samples_dim)

        rgb_map = -1 + 2 * torch.sum(
            weights * torch.sigmoid(rgb),
            self.samples_dim)  # switch to [-1,1] value range
        # st() # check rgb_map range, should be 0-255: FIXME: now it's [-1,1]

        if self.output_features:
            feature_map = torch.sum(weights * features, self.samples_dim)
        else:
            feature_map = None

        # Return surface point cloud in world coordinates.
        # This is used to generate the depth maps visualizations.
        # We use world coordinates to avoid transformation errors between
        # surface renderings from different viewpoints.
        surface_eikonal_term = None
        depth = None
        if self.return_xyz and pts is not None:
            xyz = torch.sum(weights * pts, self.samples_dim)  # B, H, W, 3
            # mask = weights[..., -1, :]  # background probability map
            depth = torch.sum(weights * z_vals.unsqueeze(-1),
                              self.samples_dim,
                              keepdim=True)  # B,h,w,1,1
            mask = (depth < 1.08).type_as(
                weights)  # * heuristic threshold for fg mask
            # mask = depth
            # mask = (depth < 1.04).type_as(weights)  # * heuristic threshold for fg mask

            # st() # todo, learn how to pass args efficiently

            # ! this is wrong, use integration normal instead?
            # ! updated 11.May, Add normalize_const here to get right results
            # normalized_xyz = xyz * normalize_const # ! normalize before sending into the network

            if return_surface_eikonal:
                xyz.requires_grad_(True)  # necessary?
                if not self.opt.use_integrated_surface_normal:
                    integrated_surface_pts = xyz.unsqueeze(self.samples_dim)
                    surface_raw = self.run_network(
                        integrated_surface_pts,
                        torch.zeros_like(integrated_surface_pts),
                        styles=kwargs['styles'])
                    surface_eikonal_term = self.get_eikonal_term(
                        integrated_surface_pts, surface_raw[..., 3:4])  # sdf
                else:  # * fixed
                    surface_eikonal_term = torch.sum(
                        weights * eikonal_term,
                        self.samples_dim).unsqueeze(-2)  # BHW13

            # st()
            # ipdb.set_trace()
        else:
            xyz = None
            mask = None

        # todo, return dict
        return rgb_map, feature_map, sdf_out, mask, xyz, eikonal_term, surface_eikonal_term, rays_d_norm, depth, dists, visibility, weights

    def sample_uniform_grid(self, batch_size: int, num_sample_inout: int,
                            device, styles):
        # todo, add raymarching

        length = self.B_MAX - self.B_MIN
        grid_random_pts = torch.rand(
            batch_size, num_sample_inout, 3,
            device=device) * length + self.B_MIN  # * uniform points in 3D
        grid_random_pts = grid_random_pts.reshape(batch_size, num_sample_inout,
                                                  1, 1, 3)  # BHWS3
        grid_random_pts_sdf = self.run_network(
            grid_random_pts, torch.zeros_like(grid_random_pts),
            styles=styles)[..., 3]

        grid_random_pts = grid_random_pts.reshape(batch_size, -1, 3)
        grid_random_pts_sdf = grid_random_pts_sdf.reshape(batch_size, -1, 1)
        valid_mask = torch.ones_like(grid_random_pts_sdf)

        return grid_random_pts, grid_random_pts_sdf, valid_mask

    def sample_near_surface_grid(self,
                                 surface_points,
                                 viewdirs,
                                 normal_stdv,
                                 styles,
                                 multiplier=1):
        """sample near surf points

        Args:
            surface_points (_type_): _description_
            viewdirs (_type_): _description_
            normal_stdv (_type_): _description_
            styles (_type_): _description_
            multiplier (int, optional): _description_. Defaults to 1.

        Returns:
            _type_: reshape to B,H,W,3(1), to apply mask 
        """
        # todo, add raymarching
        batch, h, w, _ = surface_points.shape
        perterb_norm = torch.randn_like(
            surface_points) * normal_stdv  # start with one
        uniform_points = (surface_points + perterb_norm).unsqueeze(-2)  # 4->5
        # filter out of scene range points
        # st()
        valid_pts_mask = (torch.abs(uniform_points).max(dim=-1)[0]
                          < self.opt.camera.dist_radius).int()  # B 64,64
        valid_pts_idx = torch.where(valid_pts_mask == 1)[0]
        # filtered_uniform_points = uniform_points[valid_pts_idx].reshape(1, -1, 1, 1, 3)
        # filtered_viewdirs = torch.zeros_like(filtered_uniform_points)

        uniform_points_sdf = self.run_network(uniform_points,
                                              viewdirs,
                                              styles=styles)[..., 3]

        # uniform_points = uniform_points.reshape(batch, -1, 3)
        # uniform_points_sdf = uniform_points_sdf.reshape(batch, -1, 3)

        return uniform_points, uniform_points_sdf, valid_pts_mask

    def sample_near_surface(self, surface_points, viewdirs, normal_stdv,
                            num_rand_samples):
        # todo, add raymarching
        # n,h,w,_ = surface_points.shape
        perterb_norm = torch.randn(num_rand_samples) * normal_stdv
        space_offset = perterb_norm * viewdirs
        # random sample surface points
        surf_points_u = (torch.rand(num_rand_samples, 1, 1) - 0.5) * 2
        surf_points_v = (torch.rand(num_rand_samples, 1, 1) - 0.5) * 2
        surf_grid = torch.stack([surf_points_u, surf_points_v], -1)  # N,1,1,2
        # * directly use sampled grid.
        # rand_surf_pts_sample = F.grid_sample(surface_points.permute(0,3,1,2), # N, 3, H, W
        #                              surf_grid,
        #                              mode='bilinear',
        #                              align_corners=True) # approximate surfae by interpolating  coords

        rand_surf_pts_sample = rand_surf_pts_sample.reshape(
            num_rand_samples, 3)
        return rand_surf_pts_sample + space_offset

    def sample_from_surface(self):
        pass

    def run_local_query(self, imgs, points):
        local_output = self.local_encoder(points)  # * miss img input
        # miss imgs

    def run_network_with_local_feats(self, inputs, input_dirs, styles=None):
        # to replace self.network() the global-feats only method, output raw(256+3+1)

        # input for G
        net_inputs = torch.cat([inputs, input_dirs], self.channel_dim)
        global_outputs = self.network(net_inputs, styles=styles)  # global out

        # input for L

        # fuse features

        # do rendering stuff

        # samples needed
        # todo, add local model inference
        if self.enable_local_model:
            pass

        return global_outputs

    def run_network(
            self,
            inputs,
            viewdirs,
            normalize=True,
            styles=None,
            global_only=False,
            return_sdf_only=False,  # for ray tracing
            **kwargs):
        batch_size = inputs.shape[0]

        if viewdirs.shape != inputs.shape:
            if viewdirs.ndim != inputs.ndim:
                input_dirs = viewdirs.unsqueeze(self.samples_dim)

            else:
                input_dirs = viewdirs
            input_dirs = input_dirs.expand(inputs.shape)

        else:
            input_dirs = viewdirs

        if self.z_normalize:
            normalized_pts = self.grid_warper(
                inputs)  # equivalent to normalized_pts

        net_inputs = torch.cat([normalized_pts, input_dirs],
                               self.channel_dim)  # B H W S 6
        # todo, add local model inference
        network_input = dict(net_inputs=net_inputs, styles=styles)

        def _staged_run_network(max_batch_size=50000):
            nonlocal net_inputs
            B, H, W, Steps, dim = net_inputs.shape

            net_inputs = net_inputs.reshape(B, -1, 1, 1, dim)
            # xyz = torch.empty(*net_inputs.shape[:-1],
            #                         3).to(net_inputs.device)
            outputs = torch.empty(*net_inputs.shape[:-1],
                                  4).to(net_inputs.device)

            for b in range(batch_size):
                head = 0
                batch_styles = styles[b:b + 1]
                while head < net_inputs.shape[1]:
                    tail = head + max_batch_size
                    batch_net_inputs = net_inputs[b:b + 1, head:tail, ...]
                    batch_outputs = self.network(batch_net_inputs,
                                                 batch_styles)
                    if isinstance(batch_outputs, dict):
                        outputs[b:b + 1, head:tail] = batch_outputs['raw'][...,
                                                                           3]
                        # xyz[b:b + 1, head:tail] = batch_outputs['xyz']
                    else:
                        outputs[b:b + 1, head:tail] = batch_outputs[..., 3]
                        # xyz[b:b + 1, head:tail] = batch_outputs[..., 3]
                    head += max_batch_size

            outputs = outputs.reshape(B, H, W, Steps, outputs.shape[-1])

            return outputs

        if self.enable_local_model and self.local_batch is not None and not global_only:  # only in prediction time; not sampling time
            # todo, add sampling flag?
            # assert self.local_batch is not None, 'double check input for local branch'
            current_local_batch = {
                **self.local_batch, 'world_space_pts': inputs
            }
            network_input.update(
                dict(local_data_batch=current_local_batch))  # ?

        outputs = self.network(**network_input)

        if return_sdf_only:
            return outputs[..., 3:4]

        return outputs

    def run_network_stage(self,
                          inputs,
                          viewdirs,
                          styles=None,
                          sampling_hook=False,
                          **kwargs):

        batch_size = inputs.shape[0]
        input_dirs = viewdirs.unsqueeze(self.samples_dim).expand(inputs.shape)

        if self.z_normalize:
            normalized_pts = self.grid_warper(
                inputs)  # equivalent to normalized_pts

        net_inputs = torch.cat([inputs, input_dirs], self.channel_dim)

        def _staged_run_network(max_batch_size=50000):
            nonlocal net_inputs
            B, H, W, Steps, dim = net_inputs.shape

            net_inputs = net_inputs.reshape(B, -1, 1, 1, dim)
            outputs = torch.empty(*net_inputs.shape[:-1],
                                  1).to(net_inputs.device)

            for b in range(batch_size):
                head = 0
                batch_styles = styles[b:b + 1]
                while head < net_inputs.shape[1]:
                    tail = head + max_batch_size
                    batch_net_inputs = net_inputs[b:b + 1, head:tail, ...]
                    batch_outputs = self.network(batch_net_inputs,
                                                 batch_styles)
                    if isinstance(batch_outputs, dict):
                        outputs[b:b + 1, head:tail] = batch_outputs['raw'][...,
                                                                           3]
                    else:
                        outputs[b:b + 1, head:tail] = batch_outputs[..., 3]
                    head += max_batch_size

            outputs = outputs.reshape(B, H, W, Steps, outputs.shape[-1])
            # outputs = dict(out=outputs)

            return outputs

        outputs = _staged_run_network()  # just query raw

        if return_sdf_only:
            return outputs[..., 3:4]
        return outputs

    def fuse_local_global_feat(self):
        pass

    def render_rays(
            self,
            ray_batch,
            styles=None,
            sample_mode=False,
            return_grad=False,
            return_eikonal=False,
            return_mesh=False,
            # mesh_with_shading=True,
            c2w=None,
            sample_without_grad=False,
            pts_requires_grad=False,
            **kwargs):

        batch, h, w, _ = ray_batch.shape
        split_pattern = [3, 3, 2]
        if ray_batch.shape[-1] > 8:
            split_pattern += [3]
            rays_o, rays_d, bounds, viewdirs = torch.split(
                ray_batch, split_pattern, dim=self.channel_dim)
        else:
            rays_o, rays_d, bounds = torch.split(ray_batch,
                                                 split_pattern,
                                                 dim=self.channel_dim)
            viewdirs = None

        near, far = torch.split(bounds, [1, 1], dim=self.channel_dim)

        z_vals = near * (1. - self.t_vals) + far * (self.t_vals)

        if self.perturb > 0.:
            if self.offset_sampling:
                # random offset samples
                upper = torch.cat([z_vals[..., 1:], far], -1)
                lower = z_vals.detach()
                t_rand = torch.rand(batch, h, w).unsqueeze(
                    self.channel_dim).to(z_vals.device)
            else:
                # get intervals between samples
                mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
                upper = torch.cat([mids, z_vals[..., -1:]], -1)
                lower = torch.cat([z_vals[..., :1], mids], -1)
                # stratified samples in those intervals
                t_rand = torch.rand(z_vals.shape).to(z_vals.device)

            z_vals = lower + (upper - lower) * t_rand
            # st()

        pts = rays_o.unsqueeze(self.samples_dim) + rays_d.unsqueeze(
            self.samples_dim) * z_vals.unsqueeze(
                self.channel_dim)  # * z [0.88, 1,12]

        if return_eikonal:
            if pts.requires_grad is False:
                pts.requires_grad = True

        raw = self.run_network(pts, viewdirs, styles=styles, **kwargs)
        if self.opt.return_feats:
            assert isinstance(raw, dict)
            raw, all_feats = raw['raw'], raw['all_feats']

        # viewdirs: B H W 3
        # pts: B H W S 3
        rgb_map, features, sdf, mask, xyz, eikonal_term, surface_eikonal_term, rays_d_norm, depth, dists, visibility, weights = self.volume_integration(
            raw,  # B H W S C
            z_vals,  # B H W S
            rays_d,  #  B h W 3
            pts,  # B H W 16 3
            # normalize_const=normalize_const,
            return_eikonal=return_eikonal,
            viewdirs=viewdirs,
            styles=styles,
            c2w=c2w,
            **kwargs)  # must return surface eikonal if sampling

        if self.local_batch is not None and 'sampling' in self.local_batch:
            # * not used in the cycle training
            self.local_batch.update({
                'dists': dists,
                'near': near,
                'far': far,
                'rays_d': rays_d,
                'world_space_pts': pts,
                'hit_prob': weights,
                'depth': depth
            })

        sample_batch = {
            # 'raw': raw, # !!! remove later.
            'rays_o': rays_o,
            'rays_d': rays_d,
            'dists': dists,
            'near': near,
            'far': far,
            'hit_prob': weights,
            'surface_eikonal_term': surface_eikonal_term,
            'points': pts,  # directly return 3d world space points
            'sdf': sdf,
            'gen_thumb_imgs': rgb_map,
            'features': features,
            'mask': mask,
            'xyz': xyz,
            'eikonal_term': eikonal_term,
            'depth': depth
        }
        if self.opt.return_feats:
            sample_batch.update({'all_feats': all_feats})

        if sample_without_grad:
            for k, v in sample_batch.items():
                if isinstance(v, torch.Tensor):
                    sample_batch[k] = v.detach()

        # sample around surface
        if not self.sample_mode:
            return sample_batch

        if self.opt.sample_near_surface:
            near_surface_points, uniform_points_sdf, valid_pts_mask = self.sample_near_surface_grid(
                xyz, viewdirs, self.opt.surface_sampling_stdv, styles)
            sample_batch.update({
                'points_near_surface':
                near_surface_points,
                'points_near_surface_sdf':
                uniform_points_sdf,
                'points_near_surface_valid_mask':
                valid_pts_mask
            })
        if self.opt.sample_uniform_grid:
            grid_random_pts, grid_random_pts_sdf, grid_sample_valid_mask = self.sample_uniform_grid(
                batch, self.opt.uniform_grid_sampling_num, z_vals.device,
                styles)
            sample_batch.update({
                'grid_random_pts':
                grid_random_pts,
                'grid_random_pts_sdf':
                grid_random_pts_sdf,
                'grid_sample_valid_mask':
                grid_sample_valid_mask
            })

        return sample_batch

    def query_hitting_probability_fixed_interval(self,
                                                 wd_space_pts,
                                                 ref_img_info,
                                                 return_type='weights'):
        """query the hit probability of wd space points given a ref view, sampling using fixed dist and do bilinear interp.

        Args:
            wd_space_pts (torch.Tensor): wd space points
            ref_img_info (dict): information of the reference img
            return_type (str): weights or hit probability to query

        Returns:
            torch.Tensor: hitting probability
        """
        assert return_type in ('weights', 'visibility')
        assert wd_space_pts.ndim == 5  # B H W S 3
        # assert wd_space_pts.shape[3]==1
        B, H, W, S = wd_space_pts.shape[:4]

        # get information needed
        ref_img_render_out = ref_img_info['global_render_out']
        ref_view_poses = ref_img_info['cam_settings']['poses']  # B 3 4
        ref_view_extrinsics = ref_img_info['cam_settings'][
            'extrinsics']  # B 3 4
        ref_img_styles = ref_img_info['pred_latents'][0]
        batch_size = ref_view_poses.shape[0]
        near = ref_img_render_out['near'].reshape(B, H * W, 1, 1,
                                                  1)  # B HW 1 1 1
        far = ref_img_render_out['far'].reshape(B, H * W, 1, 1,
                                                1)  # B HW 1 1 1

        # make homo for matmul
        ref_view_w2c_homo = make_homo_cam_matrices(ref_view_extrinsics)
        # ref_view_c2w_homo = make_homo_cam_matrices(ref_view_poses)

        # merge HW here as one channel, no need to merge S here for compatability with near/far shape.
        wd_space_pts = wd_space_pts.reshape(B, H * W, S, 3)
        wd_space_pts_homo = make_homo_pts(wd_space_pts).unsqueeze(
            -1)  # B HW S 4 1
        wd_space_pts = wd_space_pts.unsqueeze(-2)  # B HW S 1 3

        # wd space rays_o
        rays_o = ref_view_poses[..., 3:4].permute(0, 2, 1).reshape(
            batch_size, 1, 1, 1, 3)  # B 1 3 -> B 1 1 1 3
        # wd space rays_d
        # rays_d_wd = wd_space_pts - rays_o  # B HW S 1 3

        # transform all pts to the ref view
        ref_space_pts = ref_view_w2c_homo.reshape(
            B, 1, 1, 4, 4) @ wd_space_pts_homo  # B HW S 4 1
        # normalze back to mesh_grid like  viewdir length
        rays_d_ref = ref_space_pts[..., :3, :] / (-ref_space_pts[..., 2:3, :]
                                                  )  # B HW S 3 1

        # transform back to the wd space
        rays_d_wd = ref_view_poses.reshape(
            B, 1, 1, 3, 4)[..., :3] @ rays_d_ref  # B HW S 3 1
        rays_d_wd = rays_d_wd.permute(0, 1, 2, 4, 3)  # B HW S 1 3

        # ! get the right "near" in the ref camera space
        z_vals = near * (1. - self.t_vals) + far * (self.t_vals)  # B HW 1 1 S
        interval = (z_vals[..., 1:2] - z_vals[..., 0:1]) * rays_d_wd.norm(
            dim=-1, keepdim=True).permute(
                0, 1, 2, 4, 3)  # B HW 1 S 1, size of depth intervals
        z_vals = z_vals.permute(0, 1, 2, 4, 3)  # B HW 1 S 1

        # follow original dists
        visibility_query_ray_pts = rays_o + rays_d_wd * z_vals  # B HW S S 3

        # get the interval index pts falls in
        near_pts = visibility_query_ray_pts[..., 0:1, :]  # B HW 1 S 3
        interval_idx = (wd_space_pts - near_pts).norm(
            dim=-1, keepdim=True
        ) / interval + 1e-5  # avoid integer floating wrong, for debugging
        interval_idx_bound = torch.empty(
            B, H * W, S, 1, 2,
            device=interval_idx.device)  # for near, far of the interval index
        interval_idx_bound[...,
                           0:1] = torch.clamp(interval_idx.floor().long(),
                                              min=0,
                                              max=self.t_vals.shape[-1] - 1)
        interval_idx_bound[...,
                           1:2] = torch.clamp(interval_idx.ceil().long(),
                                              min=0,
                                              max=self.t_vals.shape[-1] - 1)
        interval_idx_bound = interval_idx_bound.long()

        # prepare
        occlusion_info = torch.empty_like(wd_space_pts[
            ..., 0:1])  # B HW S 1 1, could be visibility of hit_probability
        rays_d_wd = rays_d_wd.squeeze(
            self.samples_dim)  # remove sample dim, B HW S 3
        rays_d_ref.squeeze_(self.channel_dim)

        if self.static_viewdirs:
            viewdirs = F.normalize(rays_d_ref, dim=-1)
        else:
            viewdirs = F.normalize(rays_d_wd, dim=-1)

        z_vals = z_vals.squeeze(self.channel_dim)  # B HW S S
        # batching inference
        batching = 64**2  # * Steps
        for b in range(B):
            for step in range(0, visibility_query_ray_pts.shape[1] + batching,
                              batching):
                batch_visibility_query_ray_pts = visibility_query_ray_pts[
                    b:b + 1, step:step + batching, ...]
                batch_rays_d = rays_d_wd[b:b + 1, step:step + batching, ...]
                batch_view_dirs = viewdirs[b:b + 1, step:step + batching, ...]
                batch_ref_img_styles = ref_img_styles[b:b + 1]
                batch_ref_view_poses = ref_view_poses[b:b + 1]
                batch_z_vals = z_vals[b:b + 1, step:step + batching, ...]

                # indices
                batch_interval_idx_bound = interval_idx_bound[b:b + 1,
                                                              step:step +
                                                              batching, ...]
                batch_interval_idx = interval_idx[b:b + 1,
                                                  step:step + batching,
                                                  ...].contiguous()

                try:
                    batch_raw = self.run_network(
                        batch_visibility_query_ray_pts,
                        batch_view_dirs,
                        styles=batch_ref_img_styles,
                        global_only=True)
                except Exception as e:
                    st()
                # todo, return occlusion probability too. if the model does not converge, use the visibility. leave to the future
                batch_visibility, batch_weights = self.volume_integration(
                    batch_raw,
                    batch_z_vals,
                    # batch_rays_d,
                    batch_view_dirs,  # nput normalized viewdirs, avoid re-normalize z_vals
                    pts=None,
                    # normalize_const=normalize_const,
                    return_eikonal=False,
                    return_surface_eikonal=False,
                    viewdirs=batch_view_dirs,
                    styles=batch_ref_img_styles,
                    c2w=batch_ref_view_poses,
                    no_force_stop=True)[-2:]  # B HW S S 1

                batch_occlusion_info = {
                    'weights': batch_weights,
                    'visibility': batch_visibility
                }[return_type]

                # lerp floor and ceil value
                floor_value = torch.gather(batch_occlusion_info,
                                           self.samples_dim,
                                           batch_interval_idx_bound[..., 0:1])
                ceil_value = torch.gather(batch_occlusion_info,
                                          self.samples_dim,
                                          batch_interval_idx_bound[..., 1:2])
                lerp_weight = batch_interval_idx - batch_interval_idx_bound[
                    ..., 0:1]
                batch_weighted_occlusion_info = torch.lerp(
                    floor_value, ceil_value,
                    lerp_weight)  # lerp weight measures "end".
                # batch_hit_probs = floor_weights # for debugging

                # batch_hit_probs = batch_weights[..., -1:, :] # get the hit prob of the last element in the integration
                occlusion_info[b:b + 1, step:step + batching,
                               ...] = batch_weighted_occlusion_info

        # reshape bac to the input shape
        occlusion_info = occlusion_info.reshape(B, H, W, S, 1)
        return occlusion_info

    def query_hitting_probability_adapted_interval(self, wd_space_pts,
                                                   ref_img_info):
        """query the hit probability of wd space points given a ref view

        Args:
            wd_space_pts (torch.Tensor): wd space points
            ref_img_info (dict): information of the reference img

        Returns:
            torch.Tensor: hitting probability
        """
        assert wd_space_pts.ndim == 5  # B H W S 3
        # assert wd_space_pts.shape[3]==1
        B, H, W, S = wd_space_pts.shape[:4]

        # get information needed
        ref_img_render_out = ref_img_info['global_render_out']
        ref_view_poses = ref_img_info['cam_settings']['poses']  # B 3 4
        ref_view_extrinsics = ref_img_info['cam_settings'][
            'extrinsics']  # B 3 4
        ref_img_styles = ref_img_info['pred_latents'][0]
        batch_size = ref_view_poses.shape[0]
        # make homo for matmul
        ref_view_w2c_homo = make_homo_cam_matrices(ref_view_extrinsics)
        # ref_view_c2w_homo = make_homo_cam_matrices(ref_view_poses)

        # merge HW here as one channel, no need to merge S here for compatability with near/far shape.
        wd_space_pts = wd_space_pts.reshape(B, H * W, S, 3)
        wd_space_pts_homo = make_homo_pts(wd_space_pts).unsqueeze(
            -1)  # B HW S 4 1
        wd_space_pts = wd_space_pts.unsqueeze(-2)  # B HW S 1 3

        # wd space rays_o
        rays_o = ref_view_poses[..., 3:4].permute(0, 2, 1).reshape(
            batch_size, 1, 1, 1, 3)  # B 1 3 -> B 1 1 1 3
        # wd space rays_d
        # rays_d_wd = wd_space_pts - rays_o  # B HW S 1 3

        # transform all pts to the ref view
        ref_space_pts = ref_view_w2c_homo.reshape(
            B, 1, 1, 4, 4) @ wd_space_pts_homo  # B HW S 4 1
        # normalze back to mesh_grid like  viewdir length
        rays_d_ref = ref_space_pts[..., :3, :] / (-ref_space_pts[..., 2:3, :]
                                                  )  # B HW S 3 1
        # transform back to the wd space
        # rays_d_ref_homo = make_homo_pts(rays_d_ref[..., :3, 0]).unsqueeze(-1) # B HW S 4 1
        rays_d_wd = ref_view_poses.reshape(
            B, 1, 1, 3, 4)[..., :3] @ rays_d_ref  # B HW S 3 1
        rays_d_wd = rays_d_wd.permute(0, 1, 2, 4, 3)  # B HW S 1 3

        # ! get the right "near" in the ref camera space
        near = ref_img_render_out['near'].reshape(B, H * W, 1, 1,
                                                  1)  # B HW 1 1 1
        near_ref_pts = rays_o + rays_d_wd * near
        far = ref_space_pts[..., :3, :].norm(dim=-2,
                                             keepdim=True)  # B HW S 1 1

        # get the right near
        t_vals = torch.linspace(
            0.,
            1.,  # no need offset sampling here
            steps=self.N_samples).reshape(1, 1, 1,
                                          -1).unsqueeze(self.channel_dim).to(
                                              wd_space_pts.device)  # 1 1 1 S 1
        visibility_query_ray_pts = near_ref_pts * (
            1 - t_vals
        ) + wd_space_pts * t_vals  # interpolate between near and wd_space_pts
        z_vals = (visibility_query_ray_pts - rays_o).norm(dim=-1, keepdim=True)

        # prepare
        hit_probs = torch.empty_like(wd_space_pts[..., 0:1])  # B HW S 1 1
        rays_d_wd = rays_d_wd.squeeze(
            self.samples_dim)  # remove sample dim, B HW S 3
        rays_d_ref.squeeze_(self.channel_dim)

        if self.static_viewdirs:
            viewdirs = F.normalize(rays_d_ref, dim=-1)
        else:
            viewdirs = F.normalize(rays_d_wd, dim=-1)

        z_vals = z_vals.squeeze(self.channel_dim)  # B HW S S
        # batching inference
        batching = 64**2  # * Steps
        for b in range(B):
            for step in range(0, visibility_query_ray_pts.shape[1] + batching,
                              batching):
                batch_visibility_query_ray_pts = visibility_query_ray_pts[
                    b:b + 1, step:step + batching, ...]
                batch_rays_d = rays_d_wd[b:b + 1, step:step + batching, ...]
                batch_view_dirs = viewdirs[b:b + 1, step:step + batching, ...]
                batch_ref_img_styles = ref_img_styles[b:b + 1]
                batch_ref_view_poses = ref_view_poses[b:b + 1]
                batch_z_vals = z_vals[b:b + 1, step:step + batching, ...]

                try:
                    batch_raw = self.run_network(
                        batch_visibility_query_ray_pts,
                        batch_view_dirs,
                        styles=batch_ref_img_styles,
                        global_only=True)
                except Exception as e:
                    st()

                batch_weights = self.volume_integration(
                    batch_raw,
                    batch_z_vals,
                    # batch_rays_d,
                    batch_view_dirs,  # nput normalized viewdirs, avoid re-normalize z_vals
                    pts=None,
                    # normalize_const=normalize_const,
                    return_eikonal=False,
                    return_surface_eikonal=False,
                    viewdirs=batch_view_dirs,
                    styles=batch_ref_img_styles,
                    c2w=batch_ref_view_poses,
                    no_force_stop=True)[-1]  # B HW S S 1
                # st()
                batch_hit_probs = batch_weights[
                    ...,
                    -1:, :]  # get the hit prob of the last element in the integration
                hit_probs[b:b + 1, step:step + batching, ...] = batch_hit_probs

        # reshape bac to the input shape
        hit_probs = hit_probs.reshape(B, H, W, S, 1)
        return hit_probs

    def grid_sample_coarse_surface(self, coarse_surface=None):
        """grid-sample coarse-surface basejd on integration results. could be directly used for trainng or input of ray-marching.
        1. random sample in the 3d cube space
        2. project back to coarse_surface (interpolate)
        3. (optional) ray-marching to get accurate surface
        4. apply visible mask (todo)

        Args:
            coarse_surface (torch.Tensor, optional): integration surface. Defaults to None.
        """
        # length = self.B_MAX - self.B_MIN
        # random_points = np.random.rand(self.num_sample_inout, 3) * length + self.B_MIN # * uniform points in 3D

        # strategy now: sample on the intersection of grid.
        interp_depth = F.grid_sample(coarse_surface,
                                     self.shift_grid,
                                     mode='bilinear',
                                     align_corners=True)
        # todo
        return interp_depth

    def ray_marching(self, ray_o, ray_dirs, tau=0.005, coarse_surface=None):
        """return surface points along the ray via ray-marching (secant)

        Args:
            ray_o, ray_dirs: ray in world space
            tau (float, optional): threshold to stop raymarching. Defaults to 0.005.
            coarse_surface (_type_, optional): surface retrieved from density integration. Defaults to None.
        """

        # return rgb_map, features, sdf, mask, xyz, eikonal_term

    def forward_with_frequencies_phase_shifts(self, normalized_pts,
                                              trunc_styles, trunc_phase,
                                              viewdirs, **kwargs):
        """for compatability with pigan model

        Returns:
            dict: points features
        """
        return self.run_network(normalized_pts, viewdirs, trunc_styles,
                                **kwargs)

    def render(
            self,
            focal,
            c2w,
            near,
            far,
            styles,
            return_eikonal=False,
            return_mesh=False,
            mesh_with_shading=True,
            #    pts_requires_grad=False,
            **kwargs):
        rays_o, rays_d, viewdirs = self.get_rays(focal, c2w)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)

        # Create ray batch
        _near = near.unsqueeze(-1) * torch.ones_like(rays_d[..., :1])
        _far = far.unsqueeze(-1) * torch.ones_like(rays_d[..., :1])
        # st() # check near&far, whether can be directly applied to mesh_rays
        # pass check! near=0.88, far=1.22
        rays = torch.cat([rays_o, rays_d, _near, _far], -1)
        rays = torch.cat([rays, viewdirs], -1)
        rays = rays.float()
        # rgb, features, sdf, mask, xyz, eikonal_term =
        render_rays_out = self.render_rays(rays,
                                           styles=styles,
                                           return_eikonal=return_eikonal,
                                           **kwargs)
        if return_mesh == False:
            return {
                **render_rays_out,
                'mesh': None,
                'shading_mesh': None,
                'debug_mesh': None,
                'viewdirs': viewdirs  # b, 64, 64, 3
            }

        try:
            # st() # should enter here firsts
            # why don't need xyz?
            frostum_aligned_sdf = align_volume(render_rays_out['sdf'])
            # below _extract method is not the same as in the utils
            mesh, verts, faces = self._extract_mesh_with_marching_cubes(
                sdf=frostum_aligned_sdf)
            # todo, add shading
            shading = None
            shaded_mesh = trimesh.base.Trimesh(vertices=verts,
                                               faces=faces,
                                               vertex_colors=shading)
        # add mesh shading here
        except ValueError:
            marching_cubes_mesh = None
            shaded_mesh = None
            print('Marching cubes extraction failed.')
            print(
                'Please check whether the SDF values are all larger (or all smaller) than 0.'
            )

        return {
            **render_rays_out,
            'mesh': mesh,  # obj
            'shaded_mesh': shaded_mesh,  # obj
            'viewdirs': viewdirs  # b, 64, 64, 3
        }

        # return rgb, features, sdf, mask, xyz, eikonal_term

    def _extract_mesh_with_marching_cubes(self, sdf):
        _, h, w, d, _ = sdf.shape
        # st()
        # change coordinate order from (y,x,z) to (x,y,z)
        sdf_vol = sdf[0, ..., 0].permute(1, 0, 2).cpu().numpy()

        # scale vertices
        verts, faces, _, _ = marching_cubes(sdf_vol,
                                            0)  # * verts scale in 128^3
        # st() # check verts shape and value:
        # (37995, 3), range 4.x~128

        verts[:, 0] = (
            verts[:, 0] / float(w) - 0.5
        ) * 0.24  # * normalize back to original scene scale, [-0.12, 0.12]
        verts[:, 1] = (verts[:, 1] / float(h) - 0.5) * 0.24
        verts[:, 2] = (verts[:, 2] / float(d) - 0.5) * 0.24

        # fix normal direction
        verts[:, 2] *= -1
        verts[:, 1] *= -1
        # st() # check np or tensor, device
        mesh = trimesh.Trimesh(verts, faces)

        # verts += 1  # 0.88 - 1.12 scene scale
        return mesh, verts, faces

    def sdf_sample_pass(self,
                        cam_poses,
                        focal,
                        near,
                        far,
                        styles,
                        return_grad=False,
                        merge_spatial_dim=True):
        """sample sdf value, pts, sdf-gradient from trained 

        Returns:
            pts: torch.Tensor
            sdf: torch.Tensor
            sdf_grad: torch.Tensor
        """
        rays_o, rays_d, viewdirs = self.get_rays(focal, cam_poses)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        # normalize_const = (2 / (far - near))[0].item()

        near = near.unsqueeze(-1) * torch.ones_like(rays_d[..., :1])
        far = far.unsqueeze(-1) * torch.ones_like(rays_d[..., :1])
        z_vals = near * (1. - self.t_vals) + far * (self.t_vals)

        # get intervals between samples
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        # todo
        t_rand = torch.rand(z_vals.shape).to(z_vals.device)

        z_vals = lower + (upper - lower) * t_rand
        pts = rays_o.unsqueeze(self.samples_dim) + rays_d.unsqueeze(
            self.samples_dim) * z_vals.unsqueeze(self.channel_dim)

        if return_grad:
            pts.requires_grad_(True)

        # if self.z_normalize:
        # normalized_pts = pts * 2 / ( (far - near).unsqueeze(self.samples_dim))
        # normalized_pts = pts * 2 / (
        #     (far - near).unsqueeze(self.samples_dim))
        # else:
        #     normalized_pts = pts

        # normalized_pts to send into models
        raw = self.run_network(pts, viewdirs, styles=styles)
        _, sdf = torch.split(raw, [3, 1], dim=self.channel_dim)
        sdf = sdf.squeeze(self.channel_dim)

        if merge_spatial_dim:  # reshape to B, 3, N
            B, H, W, Steps, _ = normalized_pts.shape
            normalized_pts = normalized_pts.reshape(
                B, H * W * Steps,
                -1).permute(0, 2, 1)  # B, 3, N, for projection later
            sdf = sdf.reshape(B, 1, -1)

        return_val = {
            'points': normalized_pts,
            'sdf': sdf,
            # 'normalize_const': normalize_const
        }

        if return_grad:
            sdf_norm = self.get_eikonal_term(pts, sdf)
            if merge_spatial_dim:
                sdf_norm = sdf_norm.reshape(B, 1, -1)
            return_val.update({
                'sdf_norm': sdf_norm,
            })

        return return_val

    def mlp_init_pass(self, cam_poses, focal, near, far, styles=None):
        rays_o, rays_d, viewdirs = self.get_rays(focal, cam_poses)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)

        near = near.unsqueeze(-1) * torch.ones_like(rays_d[..., :1])
        far = far.unsqueeze(-1) * torch.ones_like(rays_d[..., :1])
        z_vals = near * (1. - self.t_vals) + far * (self.t_vals)

        # get intervals between samples
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape).to(z_vals.device)

        z_vals = lower + (upper - lower) * t_rand
        pts = rays_o.unsqueeze(self.samples_dim) + rays_d.unsqueeze(
            self.samples_dim) * z_vals.unsqueeze(self.channel_dim)

        # if self.z_normalize:
        #     normalized_pts = pts * 2 / (
        #         (far - near).unsqueeze(self.samples_dim))
        # else:
        # normalized_pts = pts

        raw = self.run_network(pts, viewdirs, styles=styles)
        _, sdf = torch.split(raw, [3, 1], dim=self.channel_dim)
        sdf = sdf.squeeze(self.channel_dim)
        target_values = pts.detach().norm(dim=-1) - ((far - near) / 4)

        return sdf, target_values

    def forward(
            self,
            cam_poses,
            focal,
            near,
            far,
            styles=None,
            return_eikonal=False,
            geometry_sample=None,
            return_surface_eikonal=False,
            local_data_batch=None,  # sampled from synthetic dataset
            sample_mode=False,
            return_mesh=False,
            mesh_with_shading=True,
            return_sdf_only=False,
            # cache_mlpout=False,
            **kwargs):

        self.sample_mode = sample_mode

        # if not kwargs.get('sample_mode', False) and self.enable_local_model:
        if self.enable_local_model:  # * deal with L branch
            # None input if input is z-space code
            # todo, update assert, must enable local when input is not z-space normal
            # todo, don't share the same forward() for all functions

            if local_data_batch is not None:
                if 'gen_imgs' in local_data_batch:
                    assert local_data_batch['gen_imgs'].shape[
                        -1] == 256, 'resize res condition'
                self.local_batch = local_data_batch
            else:
                # assert sample_mode or inference_mode
                self.local_batch = None
        else:
            self.local_batch = None

        render_out = self.render(
            focal,
            c2w=cam_poses,
            near=near,
            far=far,
            styles=styles,
            return_eikonal=return_eikonal,
            return_surface_eikonal=return_surface_eikonal,
            return_mesh=return_mesh,
            mesh_with_shading=mesh_with_shading,
            return_sdf_only=return_sdf_only,
            # sample_mode=sample_mode,
            **kwargs)

        if geometry_sample:  # has 3d supervision
            sample_corpus = ['uniform_pts']

            if geometry_sample['xyz'] is not None:
                sample_corpus += ['xyz']

            if return_surface_eikonal and geometry_sample['xyz'] is not None:
                geometry_sample['xyz'].requires_grad_(True)

            for k in sample_corpus:
                if k not in geometry_sample:
                    continue
                samples = geometry_sample[k]
                if samples.ndim == 4:  # surface points, add step dim
                    samples = samples.unsqueeze(self.samples_dim)  # B,H,W,1,3

                # ! add normalization const
                # samples = samples * geometry_sample[
                #     'normalize_const']  # ! bugs in surface from here.todo reflection
                input_points_out_raw = self.run_network(
                    samples,  # todo, 
                    # render_out['viewdirs'],
                    torch.zeros_like(
                        samples
                    ),  # viewdirs do not enage in geometry calculation
                    styles=styles)
                input_points_sdf = input_points_out_raw[..., 3:4]
                render_out[f'{k}_rec'] = input_points_sdf  # B H W Steps 1

                if return_surface_eikonal and k == 'xyz':
                    render_out[
                        f'{k}_rec_eikonal_term'] = self.get_eikonal_term(
                            samples,
                            input_points_sdf)  # normal for surface points

            # todo, add eikonal?

        if sample_mode:
            render_out = self.collate_fn(render_out)
        else:

            if render_out['xyz'] != None:
                render_out['xyz'] = render_out['xyz'].permute(0, 3, 1,
                                                              2).contiguous()
                render_out['mask'] = render_out['mask'].permute(
                    0, 4, 1, 2, 3).contiguous()

        # shared operation
        render_out['gen_thumb_imgs'] = render_out['gen_thumb_imgs'].permute(
            0, 3, 1, 2).contiguous()
        if self.output_features:
            render_out['features'] = render_out['features'].permute(
                0, 3, 1, 2).contiguous()

        if self.sample_mode:
            self.sample_mode = False  # flip flag
        return render_out

        # return rgb, features, sdf, mask, xyz, eikonal_term

    def collate_fn(self, render_out, match_inference_dim=True):
        # merge sampled uniform points(surface and uniform sampled)

        B = render_out['gen_thumb_imgs'].shape[0]
        uniform_points = torch.empty(
            B, 0, 3, device=render_out['gen_thumb_imgs'].device)
        uniform_points_sdf = torch.empty(
            B, 0, 1, device=render_out['gen_thumb_imgs'].device)
        uniform_points_valid_mask = torch.empty(
            B, 0, 1, device=render_out['gen_thumb_imgs'].device)

        # sample around surface
        if self.opt.sample_near_surface:
            points_near_surface, points_near_surface_sdf, points_near_surface_valid_mask = [
                render_out[k] for k in [
                    'points_near_surface', 'points_near_surface_sdf',
                    'points_near_surface_valid_mask'
                ]
            ]
            uniform_points = torch.cat(
                [uniform_points,
                 points_near_surface.reshape(B, -1, 3)], dim=1)
            uniform_points_sdf = torch.cat([
                uniform_points_sdf,
                points_near_surface_sdf.reshape(B, -1, 1)
            ],
                                           dim=1)
            uniform_points_valid_mask = torch.cat([
                uniform_points_valid_mask,
                points_near_surface_valid_mask.reshape(B, -1, 1)
            ],
                                                  dim=1)
        if self.opt.sample_uniform_grid:
            grid_random_pts, grid_random_pts_sdf, grid_sample_valid_mask = [
                render_out[k] for k in [
                    'grid_random_pts', 'grid_random_pts_sdf',
                    'grid_sample_valid_mask'
                ]
            ]

            uniform_points = torch.cat(
                [uniform_points,
                 grid_random_pts.reshape(B, -1, 3)], dim=1)
            uniform_points_sdf = torch.cat(
                [uniform_points_sdf,
                 grid_random_pts_sdf.reshape(B, -1, 1)],
                dim=1)
            uniform_points_valid_mask = torch.cat([
                uniform_points_valid_mask,
                grid_sample_valid_mask.reshape(B, -1, 1)
            ],
                                                  dim=1)

        if match_inference_dim:  # match sample_dim, run_network needs 5d input
            uniform_points = uniform_points.reshape(B, -1, 1, 1, 3)
            uniform_points_sdf = uniform_points_sdf.reshape(B, -1, 1, 1, 1)
            uniform_points_valid_mask = uniform_points_valid_mask.reshape(
                B, -1, 1, 1, 1)

        render_out.update(
            {
                'uniform_pts': uniform_points,
                'uniform_points_sdf':  # todo, rename
                uniform_points_sdf,
                'uniform_points_valid_mask': uniform_points_valid_mask
            })

        return render_out
