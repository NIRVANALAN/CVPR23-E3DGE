from numpy import deprecate
import torch

from project.models.op import grid_sample_gradfix


@deprecate
def grid_sample_differientiable(image, optical, **kwargs):
    N, C, IH, IW = image.shape
    _, H, W, _ = optical.shape

    ix = optical[..., 0]
    iy = optical[..., 1]

    ix = ((ix + 1) / 2) * (IW - 1)
    iy = ((iy + 1) / 2) * (IH - 1)
    with torch.no_grad():
        ix_nw = torch.floor(ix)
        iy_nw = torch.floor(iy)
        ix_ne = ix_nw + 1
        iy_ne = iy_nw
        ix_sw = ix_nw
        iy_sw = iy_nw + 1
        ix_se = ix_nw + 1
        iy_se = iy_nw + 1

    nw = (ix_se - ix) * (iy_se - iy)
    ne = (ix - ix_sw) * (iy_sw - iy)
    sw = (ix_ne - ix) * (iy - iy_ne)
    se = (ix - ix_nw) * (iy - iy_nw)

    with torch.no_grad():
        torch.clamp(ix_nw, 0, IW - 1, out=ix_nw)
        torch.clamp(iy_nw, 0, IH - 1, out=iy_nw)

        torch.clamp(ix_ne, 0, IW - 1, out=ix_ne)
        torch.clamp(iy_ne, 0, IH - 1, out=iy_ne)

        torch.clamp(ix_sw, 0, IW - 1, out=ix_sw)
        torch.clamp(iy_sw, 0, IH - 1, out=iy_sw)

        torch.clamp(ix_se, 0, IW - 1, out=ix_se)
        torch.clamp(iy_se, 0, IH - 1, out=iy_se)

    image = image.reshape(N, C, IH * IW)

    nw_val = torch.gather(image, 2, (iy_nw * IW + ix_nw).long().reshape(
        N, 1, H * W).repeat(1, C, 1))
    ne_val = torch.gather(image, 2, (iy_ne * IW + ix_ne).long().reshape(
        N, 1, H * W).repeat(1, C, 1))
    sw_val = torch.gather(image, 2, (iy_sw * IW + ix_sw).long().reshape(
        N, 1, H * W).repeat(1, C, 1))
    se_val = torch.gather(image, 2, (iy_se * IW + ix_se).long().reshape(
        N, 1, H * W).repeat(1, C, 1))

    out_val = (nw_val.reshape(N, C, H, W) * nw.reshape(N, 1, H, W) +
               ne_val.reshape(N, C, H, W) * ne.reshape(N, 1, H, W) +
               sw_val.reshape(N, C, H, W) * sw.reshape(N, 1, H, W) +
               se_val.reshape(N, C, H, W) * se.reshape(N, 1, H, W))

    return out_val


def index(feat, uv):
    '''

    :param feat: [B, C, H, W] image features
    :param uv: [B, 2, N] uv coordinates in the image plane, range [-1, 1]
    :return: [B, C, N] image features at the uv coordinates
    '''
    uv = uv.transpose(1, 2)  # [B, N, 2]
    uv = uv.unsqueeze(2)  # [B, N, 1, 2]
    # NOTE: for newer PyTorch, it seems that training results are degraded due to implementation diff in F.grid_sample
    # for old versions, simply remove the aligned_corners argument.

    # samples = torch.nn.functional.grid_sample(feat, uv)  # [B, C, N, 1]
    samples = grid_sample_gradfix.grid_sample(
        feat, uv)  # allow high order diff, to calculate eikonal term
    # samples = grid_sample_differientiable(feat, uv)  # to calculate eikonal term. [B, C, N, 1]. deprecated due to overflow and speed
    return samples[:, :, :, 0]  # [B, C, N]


def orthogonal(points, calibrations, transforms=None):
    '''
    Compute the orthogonal projections of 3D points into the image plane by given projection matrix
    :param points: [B, 3, N] Tensor of 3D points
    :param calibrations: [B, 4, 4] Tensor of projection matrix
    :param transforms: [B, 2, 3] Tensor of image transform matrix
    :return: xyz: [B, 3, N] Tensor of xyz coordinates in the image plane
    '''
    rot = calibrations[:, :3, :3]
    trans = calibrations[:, :3, 3:4]
    pts = torch.baddbmm(trans, rot, points)  # [B, 3, N]
    if transforms is not None:
        scale = transforms[:2, :2]
        shift = transforms[:2, 2:3]
        pts[:, :2, :] = torch.baddbmm(shift, scale, pts[:, :2, :])
    return pts


def perspective(points, calibrations, transforms=None):
    '''
    Compute the perspective projections of 3D points into the image plane by given projection matrix
    :param points: [Bx3xN] Tensor of 3D points
    :param calibrations: [Bx4x4] Tensor of projection matrix
    :param transforms: [Bx2x3] Tensor of image transform matrix
    :return: xy: [Bx2xN] Tensor of xy coordinates in the image plane
    '''
    # 11.May all validated here
    rot = calibrations[:, :3, :3]
    trans = calibrations[:, :3, 3:4]
    homo = torch.baddbmm(
        trans, rot, points
    )  # [B, 3, N] # * homo coordinates in canonical camera space, 0.88 - 1.12

    if homo[0, -1, 0] < 0:  # look at -z
        z = homo[:, 2:3, :] * -1
    else:
        z = homo[:, 2:3, :]

    # xy = homo[:, :2, :] / homo[:, 2:3, :]
    xy = homo[:, :2, :] / z
    if transforms is not None:
        scale = transforms[:2, :2]
        shift = transforms[:2, 2:3]
        xy = torch.baddbmm(shift, scale, xy)

    xyz = torch.cat([xy, z], 1)  # xy(img-space) + depth
    return xyz


class render_functions():

    def __init__(self) -> None:
        pass

    def run_Secant_method(self,
                          f_low,
                          f_high,
                          z_low,
                          z_high,
                          n_secant_steps,
                          ray_origins,
                          ray_directions,
                          z,
                          logit_tau,
                          tohalf=False,
                          truncated_frequencies=None,
                          truncated_phase_shifts=None):
        ''' Runs the secant method for interval [z_low, z_high].

        Args:
            z_low (tensor): start values for the interval
            z_high (tensor): end values for the interval
            n_secant_steps (int): number of steps
            ray_origins (tensor): ray start points
            ray_directions (tensor): ray direction vectors
            z (tensor): latent conditioned code z
            logit_tau (float): threshold value in probs
        '''
        z_pred = -f_low * (z_high - z_low) / (f_high - f_low) + z_low
        for i in range(n_secant_steps):
            p_mid = ray_origins + z_pred.unsqueeze(-1) * ray_directions
            with torch.no_grad():
                if truncated_frequencies is not None and truncated_phase_shifts is not None:
                    rgb_sigma = self.siren.forward_with_frequencies_phase_shifts(
                        p_mid.unsqueeze(1),
                        truncated_frequencies,
                        truncated_phase_shifts,
                        ray_directions=ray_directions.unsqueeze(1))
                else:
                    rgb_sigma = self.siren(
                        p_mid.unsqueeze(1),
                        z,
                        ray_directions=ray_directions.unsqueeze(1))
                if tohalf:
                    f_mid = rgb_sigma[..., -1].half().squeeze(1) - logit_tau
                else:
                    f_mid = rgb_sigma[..., -1].squeeze(1) - logit_tau
            inz_low = f_mid < 0
            if inz_low.sum() > 0:
                z_low[inz_low] = z_pred[inz_low]
                f_low[inz_low] = f_mid[inz_low]
            if (inz_low == 0).sum() > 0:
                z_high[inz_low == 0] = z_pred[inz_low == 0]
                f_high[inz_low == 0] = f_mid[inz_low == 0]

            z_pred = -f_low * (z_high - z_low) / (f_high - f_low) + L

        return z_pred.data

    def perform_ray_marching(
        self,
        rgb_sigma,
        z_vals,
        ray_origins,
        ray_directions,
        n_steps,
        n_samples,
        interval,
        z=None,
        tau=0.5,
        n_secant_steps=8,
        depth_range=[0.88, 1.12],
        method='secant',
        clamp_mode='relu',
        tohalf=False,
        truncated_frequencies=None,
        truncated_phase_shifts=None,
    ):
        ''' Performs ray marching to detect surface points.

        The function returns the surface points as well as z_i of the formula
            ray(z_i) = ray_origins + z_i * ray_directions
        which hit the surface points. In addition, masks are returned for
        illegal values.

        Args:
            rgb_sigma: the output of siren network
            ray_origins (tensor): ray start points of dimension B x N x 3
            ray_directions (tensor):ray direction vectors of dim B x N x 3
            interval: sampling interval
            z (tensor): latent conditioned code
            tau (float): threshold value
            n_secant_steps (int): number of secant refinement steps
            depth_range (tuple): range of possible depth values (not relevant when
                using cube intersection)
            method (string): refinement method (default: secant)
        '''
        # n_pts = W * H
        batch_size, n_pts, D = ray_origins.shape
        device = self.device
        if tohalf:
            logit_tau = torch.from_numpy(
                get_logits_from_prob(tau)[np.newaxis].astype(
                    np.float16)).to(device)
        else:
            logit_tau = torch.from_numpy(
                get_logits_from_prob(tau)[np.newaxis].astype(
                    np.float32)).to(device)

        alphas = rgb_sigma[..., -1] - logit_tau
        #print("alphas shape:", alphas.shape)

        # Create mask for valid points where the first point is not occupied
        mask_0_not_occupied = alphas[:, :, 0] < 0

        # Calculate if sign change occurred and concat 1 (no sign change) in
        # last dimension
        sign_matrix = torch.cat([
            torch.sign(alphas[:, :, :-1] * alphas[:, :, 1:]),
            torch.ones(batch_size, n_pts, 1).to(device)
        ],
                                dim=-1)
        cost_matrix = sign_matrix * torch.arange(n_steps, 0,
                                                 -1).float().to(device)
        # Get first sign change and mask for values where a.) a sign changed
        # occurred and b.) no a neg to pos sign change occurred (meaning from
        # inside surface to outside)
        values, indices = torch.min(cost_matrix, -1)
        mask_sign_change = values < 0
        mask_neg_to_pos = alphas[torch.arange(batch_size).unsqueeze(-1),
                                 torch.arange(n_pts).unsqueeze(-0),
                                 indices] < 0

        # Define mask where a valid depth value is found
        mask = mask_sign_change & mask_neg_to_pos & mask_0_not_occupied

        # Get depth values and function values for the interval
        # to which we want to apply the Secant method
        n = batch_size * n_pts
        z_low = z_vals.reshape(n, n_steps, 1)[torch.arange(n),
                                              indices.reshape(n)].reshape(
                                                  batch_size, n_pts)[mask]
        f_low = alphas.reshape(n, n_steps, 1)[torch.arange(n),
                                              indices.reshape(n)].reshape(
                                                  batch_size, n_pts)[mask]
        indices = torch.clamp(indices + 1, max=n_steps - 1)
        z_high = z_vals.reshape(n, n_steps, 1)[torch.arange(n),
                                               indices.reshape(n)].reshape(
                                                   batch_size, n_pts)[mask]
        f_high = alphas.reshape(n, n_steps, 1)[torch.arange(n),
                                               indices.reshape(n)].reshape(
                                                   batch_size, n_pts)[mask]

        ray_origins_masked = ray_origins[mask]
        ray_direction_masked = ray_directions[mask]

        # write z in pointwise format

        if z is not None and z.shape[-1] != 0:
            z = z.unsqueeze(1).repeat(1, n_pts, 1)[mask]
        if truncated_frequencies is not None and truncated_phase_shifts is not None:
            truncated_frequencies = truncated_frequencies.unsqueeze(1).repeat(
                1, n_pts, 1)[mask]
            truncated_phase_shifts = truncated_phase_shifts.unsqueeze(
                1).repeat(1, n_pts, 1)[mask]

        # Apply surface depth refinement step (e.g. Secant method)
        if method == 'secant' and mask.sum() > 0:
            d_pred = self.run_Secant_method(
                f_low.clone(),
                f_high.clone(),
                z_low.clone(),
                z_high.clone(),
                n_secant_steps,
                ray_origins_masked,
                ray_direction_masked,
                z,
                logit_tau,
                tohalf=tohalf,
                truncated_frequencies=truncated_frequencies,
                truncated_phase_shifts=truncated_phase_shifts)
        elif method == 'bisection' and mask.sum() > 0:
            d_pred = self.run_Bisection_method(
                z_low.clone(),
                z_high.clone(),
                n_secant_steps,
                ray_origins_masked,
                ray_direction_masked,
                z,
                logit_tau,
                tohalf=tohalf,
                truncated_frequencies=truncated_frequencies,
                truncated_phase_shifts=truncated_phase_shifts)
        else:
            d_pred = torch.ones(ray_direction_masked.shape[0]).to(device)

        # for sanity
        d_pred_out = torch.ones(batch_size, n_pts).to(device)
        d_pred_out[mask] = d_pred
        # sample points
        ray_start = torch.ones(batch_size, n_pts).to(device) * depth_range[0]
        ray_end = torch.ones(batch_size, n_pts).to(device) * depth_range[1]
        ray_start_masked = d_pred - interval
        ray_end_masked = d_pred + interval
        # in case of cross the near boundary
        mask_cross_near_bound = ray_start_masked < depth_range[0]
        ray_start_masked[mask_cross_near_bound] = depth_range[0]
        ray_end_masked[mask_cross_near_bound] = depth_range[0] + interval * 2
        # in case of cross the far boundary
        mask_cross_far_bound = ray_end_masked > depth_range[1]
        ray_end_masked[mask_cross_far_bound] = depth_range[1]
        ray_start_masked[mask_cross_far_bound] = depth_range[1] - interval * 2
        # for sanity
        ray_start[mask] = ray_start_masked
        ray_end[mask] = ray_end_masked

        # pred_z_vals shape: [B, n_pts, n_samples]
        z_vals_init = torch.linspace(0, 1, n_samples, device=device)
        pred_z_vals = ray_start.unsqueeze(
            -1) + (ray_end - ray_start).unsqueeze(-1) * z_vals_init

        return pred_z_vals.unsqueeze(-1), d_pred_out, mask

    def forward(self,
                z,
                img_size,
                fov,
                ray_start,
                ray_end,
                h_stddev,
                v_stddev,
                h_mean,
                v_mean,
                hierarchical_sample,
                sample_dist=None,
                lock_view_dependence=False,
                **kwargs):
        """
        Generates images from a noise vector, rendering parameters, and camera distribution.
        Uses the hierarchical sampling scheme described in NeRF.
        """

        batch_size = z.shape[0]
        outputs = {}

        all_outputs = []
        all_z_vals = []

        # short hand
        pred_occ = kwargs.get('pred_occ', False)
        surface_sample = kwargs.get('surface_sample', False)
        with_normal_loss = kwargs.get('with_normal_loss', False)
        with_tvprior = kwargs.get('with_tvprior', False)
        with_opacprior = kwargs.get('with_opacprior', False)
        num_steps_coarse = kwargs.get('num_steps_coarse', 9)
        num_steps_surface = kwargs.get('num_steps_surface', 9)
        num_steps_fine = kwargs.get('num_steps_fine', 6)

        # Generate initial camera rays and sample points.
        with torch.no_grad():
            points_cam, z_vals, rays_d_cam = get_initial_rays_trig(
                batch_size,
                num_steps_coarse,
                resolution=(img_size, img_size),
                device=self.device,
                fov=fov,
                ray_start=ray_start,
                ray_end=ray_end)  # batch_size, pixels, num_steps_coarse, 1
            transformed_points, z_vals, transformed_ray_directions, transformed_ray_origins, pitch, yaw = transform_sampled_points(
                points_cam,
                z_vals,
                rays_d_cam,
                h_stddev=h_stddev,
                v_stddev=v_stddev,
                h_mean=h_mean,
                v_mean=v_mean,
                device=self.device,
                mode=sample_dist)
            transformed_points = transformed_points.reshape(
                batch_size, img_size * img_size * num_steps_coarse, 3)

        # Model prediction on course points
        if not surface_sample:
            # transformed_points, transformed_ray_directions_expanded, coarse_output: torch.float16; z: torch.float32
            coarse_output = self.siren(
                transformed_points,
                z,
                ray_directions=self._expand_ray_directions(
                    transformed_ray_directions, num_steps_coarse,
                    lock_view_dependence))
            coarse_output = coarse_output.reshape(batch_size,
                                                  img_size * img_size,
                                                  num_steps_coarse, 4)
            all_outputs = [coarse_output]
            all_z_vals = [z_vals]
        else:
            with torch.no_grad():
                coarse_output = self.siren(
                    transformed_points,
                    z,
                    ray_directions=self._expand_ray_directions(
                        transformed_ray_directions, num_steps_coarse,
                        lock_view_dependence))
                coarse_output = coarse_output.reshape(batch_size,
                                                      img_size * img_size,
                                                      num_steps_coarse, 4)

                # surface_z_vals shape: [batch_size, num_pixels**2, num_steps_surface, 1]
                surface_z_vals, pred_depth, mask = self.perform_ray_marching(
                    coarse_output.detach(),
                    z_vals,
                    transformed_ray_origins,
                    transformed_ray_directions,
                    n_steps=num_steps_coarse,
                    n_samples=num_steps_surface,
                    z=z,
                    interval=kwargs['interval'],
                    n_secant_steps=kwargs.get('n_secant_steps', 8),
                    method=kwargs.get('method', 'secant'),
                    tau=kwargs.get('tau', 0.5),
                    tohalf=True)
                surface_z_vals = surface_z_vals.detach()
                surface_around_points = transformed_ray_origins.unsqueeze(
                    2).contiguous() + transformed_ray_directions.unsqueeze(
                        2).contiguous() * surface_z_vals.expand(
                            -1, -1, -1, 3).contiguous()
                surface_around_points, surface_z_vals = perturb_points(
                    surface_around_points, surface_z_vals,
                    transformed_ray_directions, self.device)
                surface_around_points = surface_around_points.reshape(
                    batch_size, -1, 3)
                outputs['mask'] = mask
                outputs['pred_depth'] = pred_depth.detach().cpu().clamp(
                    min=ray_start, max=ray_end)
