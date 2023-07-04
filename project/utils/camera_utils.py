import numpy as np
from pytorch3d.renderer import FoVPerspectiveCameras, look_at_view_transform
import torch
from torch.nn import functional as F


################# Camera parameters sampling ####################
def generate_camera_params(
    resolution,
    device,
    batch=1,
    locations=None,
    sweep=False,
    uniform=False,
    azim_range=0.3,
    elev_range=0.15,
    fov_ang=6,
    dist_radius=0.12,
    return_calibs=False,
    azim_mean=0.,
    elev_mean=0.,
):
    if locations != None:
        azim = locations[:, 0].reshape(-1, 1)
        elev = locations[:, 1].reshape(-1, 1)

        # generate intrinsic parameters
        # fix distance to 1
        dist = torch.ones(azim.shape[0], 1, device=device)
        near, far = (dist -
                     dist_radius).unsqueeze(-1), (dist +
                                                  dist_radius).unsqueeze(-1)
        fov_angle = fov_ang * torch.ones(
            azim.shape[0], 1, device=device).reshape(-1, 1) * np.pi / 180
        focal = 0.5 * resolution / torch.tan(fov_angle).unsqueeze(-1)
    elif sweep:
        # generate camera locations on the unit sphere
        azim = (-azim_range +
                (2 * azim_range / 7) * torch.arange(8, device=device)).reshape(
                    -1, 1).repeat(batch, 1)
        elev = (
            -elev_range + 2 * elev_range *
            torch.rand(batch, 1, device=device).repeat(1, 8).reshape(-1, 1))

        # generate intrinsic parameters
        dist = (torch.ones(batch, 1, device=device)).repeat(1,
                                                            8).reshape(-1, 1)
        near, far = (dist -
                     dist_radius).unsqueeze(-1), (dist +
                                                  dist_radius).unsqueeze(-1)
        fov_angle = fov_ang * torch.ones(batch, 1, device=device).repeat(
            1, 8).reshape(-1, 1) * np.pi / 180
        focal = 0.5 * resolution / torch.tan(fov_angle).unsqueeze(-1)
    else:
        # sample camera locations on the unit sphere
        if uniform:
            azim = (-azim_range +
                    2 * azim_range * torch.rand(batch, 1, device=device))
            elev = (-elev_range +
                    2 * elev_range * torch.rand(batch, 1, device=device))
        else:
            azim = (azim_range * torch.randn(batch, 1, device=device))
            elev = (elev_range * torch.randn(batch, 1, device=device))

        # generate intrinsic parameters
        dist = torch.ones(
            batch, 1,
            device=device)  # restrict camera position to be on the unit sphere
        near, far = (dist -
                     dist_radius).unsqueeze(-1), (dist +
                                                  dist_radius).unsqueeze(
                                                      -1)  # 0.88 - 1.12
        fov_angle = fov_ang * torch.ones(
            batch, 1, device=device) * np.pi / 180  # full fov is 12 degrees
        focal = 0.5 * resolution / torch.tan(fov_angle).unsqueeze(-1)

    azim = azim_mean + azim
    elev = elev_mean + elev

    viewpoint = torch.cat([azim, elev], 1)

    #### Generate camera extrinsic matrix ##########

    # convert angles to xyz coordinates
    x = torch.cos(elev) * torch.sin(azim)
    y = torch.sin(elev)
    z = torch.cos(elev) * torch.cos(azim)
    camera_dir = torch.stack([x, y, z], dim=1).reshape(-1, 3)
    camera_loc = dist * camera_dir

    # get rotation matrices (assume object is at the world coordinates origin)
    up = torch.tensor([[0, 1, 0]]).float().to(device) * torch.ones_like(dist)
    z_axis = F.normalize(camera_dir,
                         eps=1e-5)  # the -z direction points into the screen
    x_axis = F.normalize(torch.cross(up, z_axis, dim=1), eps=1e-5)
    y_axis = F.normalize(torch.cross(z_axis, x_axis, dim=1), eps=1e-5)
    is_close = torch.isclose(x_axis, torch.tensor(0.0),
                             atol=5e-3).all(dim=1, keepdim=True)
    if is_close.any():
        replacement = F.normalize(torch.cross(y_axis, z_axis, dim=1), eps=1e-5)
        x_axis = torch.where(is_close, replacement, x_axis)
    w2c_R = torch.cat(
        (x_axis[:, None, :], y_axis[:, None, :], z_axis[:, None, :]),
        dim=1)  # cat in rows. ugly code.
    c2w_R = w2c_R.transpose(1, 2)  # transpose into columms.

    # R = torch.cat((x_axis[:, :, None], y_axis[:, :, None], z_axis[:, :, None]),
    #               dim=-1) # above code equivalent to directly cat in columns

    T = camera_loc[:, :, None]  # B 3 1
    poses = torch.cat((c2w_R, T), -1)  # * B, 3, 4, c2w matrix

    if return_calibs:
        T_extrinsics = -w2c_R @ T  # Bx3x3 @ Bx3x1
        extrinsics = torch.cat((w2c_R, T_extrinsics),
                               dim=-1)  # * Bx3x4, w2c matrix
        # homo_E = torch.cat((extrinsics, torch.zeros(batch, 1, 4, device=device)), 1)
        # homo_E[:, 3, 3] = 1
        if sweep:
            batch *= 8  # 8 sweep views

        focal_mat = torch.zeros(batch, 3, 2, device=device)
        focal_mat[:, 0, 0] = focal_mat[:, 1, 1] = focal[0].squeeze()

        offset_mat = torch.ones(batch, 3, 1, device=device)
        offset_mat[:, 0, -1] = offset_mat[:, 1, -1] = 0.5 * resolution

        # pix_space_intrinsics = torch.cat([focal_mat, offset_mat], -1) # B, 3, 3. This is pixel-space intrinsics
        uv_offset_mat = torch.zeros_like(offset_mat)
        uv_offset_mat[:, -1, -1] = 1.  # homo-coord
        uv_space_intrinsics = torch.cat(
            [focal_mat / (resolution / 2), uv_offset_mat
             ],  # 2f/w, map pixel space to [-1,1]
            -1)  # B, 3, 3. map 3D points to [-1,1]
        intrinsics = uv_space_intrinsics
        calibs = intrinsics @ extrinsics  # B,3,4
        homo_coord = torch.cat(
            [torch.zeros(batch, 1, 3),
             torch.ones(batch, 1, 1)], dim=-1).to(device)  # B, 1, 4
        calibs_homo = torch.cat([calibs, homo_coord], -2)  # B, 4, 4
        return dict(
            poses=poses,
            extrinsics=extrinsics,
            focal=focal,
            near=near,
            far=far,
            viewpoint=viewpoint,
            intrinsics=intrinsics,
            calibs=calibs_homo,
            locations=locations,
            azim_range=azim_range,
            elev_range=elev_range,
        )

    return poses, focal, near, far, viewpoint


def create_cameras(R=None,
                   T=None,
                   azim=0,
                   elev=0.,
                   dist=1.,
                   fov=12.,
                   znear=0.01,
                   device="cuda") -> FoVPerspectiveCameras:
    """
    all the camera parameters can be a single number, a list, or a torch tensor.
    """
    if R is None or T is None:
        R, T = look_at_view_transform(dist=dist,
                                      azim=azim,
                                      elev=elev,
                                      device=device)
    cameras = FoVPerspectiveCameras(device=device,
                                    R=R,
                                    T=T,
                                    znear=znear,
                                    fov=fov)
    return cameras


def make_homo_cam_matrices(calibs: torch.Tensor):
    assert calibs.shape[-2:] == torch.Size((3, 4))
    homo_calibs = torch.cat((calibs, torch.zeros_like(calibs[..., 0:1, :])),
                            dim=-2)
    homo_calibs[..., -1, -1] = 1
    return homo_calibs


def make_homo_pts(pts: torch.Tensor):
    assert pts.shape[-1] == 3
    homo_pts = torch.cat((pts, torch.ones_like(pts[..., 0:1])), dim=-1)
    return homo_pts
