import numpy as np
from torch import nn
from pytorch3d.renderer import (FoVPerspectiveCameras, MeshRasterizer,
                                MeshRenderer, PointLights,
                                RasterizationSettings, SoftPhongShader,
                                TexturesVertex)
import torch
import trimesh
from scipy.spatial import Delaunay
from skimage.measure import marching_cubes
from torch.nn import functional as F
from pytorch3d.structures import Meshes


#################### Mesh generation util functions ########################
# Reshape sampling volume to camera frostum
def align_volume(volume, near=0.88, far=1.12):  # *todo
    # st()
    b, h, w, d, c = volume.shape
    yy, xx, zz = torch.meshgrid(torch.linspace(-1, 1, h),
                                torch.linspace(-1, 1, w),
                                torch.linspace(-1, 1, d))

    grid = torch.stack([xx, yy, zz], -1).to(volume.device)

    frostum_adjustment_coeffs = torch.linspace(far / near, 1,
                                               d).reshape(1, 1, 1, -1,
                                                          1).to(volume.device)
    frostum_grid = grid.unsqueeze(0)
    frostum_grid[..., :2] = frostum_grid[..., :2] * frostum_adjustment_coeffs
    out_of_boundary = torch.any(
        (frostum_grid.lt(-1).logical_or(frostum_grid.gt(1))), -1, keepdim=True)
    frostum_grid = frostum_grid.permute(0, 3, 1, 2, 4).contiguous()
    permuted_volume = volume.permute(0, 4, 3, 1, 2).contiguous()
    final_volume = F.grid_sample(permuted_volume,
                                 frostum_grid,
                                 padding_mode="border",
                                 align_corners=True)
    final_volume = final_volume.permute(0, 3, 4, 2, 1).contiguous()
    # set a non-zero value to grid locations outside of the frostum to avoid marching cubes distortions.
    # It happens because pytorch grid_sample uses zeros padding.
    final_volume[out_of_boundary] = 1

    return final_volume


# Extract mesh with marching cubes
def extract_mesh_with_marching_cubes(sdf, shading=False):
    _, h, w, d, _ = sdf.shape

    # change coordinate order from (y,x,z) to (x,y,z)
    sdf_vol = sdf[0, ..., 0].permute(1, 0, 2).cpu().numpy()

    # scale vertices
    verts, faces, _, _ = marching_cubes(sdf_vol, 0)  # * verts scale in 128^3
    # st() # check verts shape and value: (37xxx, 3) of variable len, min~max: 0~128

    verts[:, 0] = (
        verts[:, 0] / float(w) -
        0.5) * 0.24  # * normalize back to original scene scale, [-0.12, 0.12]
    verts[:, 1] = (verts[:, 1] / float(h) - 0.5) * 0.24
    verts[:, 2] = (verts[:, 2] / float(d) - 0.5) * 0.24

    # fix normal direction
    verts[:, 2] *= -1
    verts[:, 1] *= -1
    mesh = trimesh.Trimesh(verts, faces)

    return mesh


# def _mesh_render(
#                focal,
#                c2w,
#                near,
#                far,
#                styles,
#                return_eikonal=False,
#                **kwargs):
#         # 1. now assume it in world coord, need further check TODO
#         st()
#         rays_o, rays_d, viewdirs = mesh_get_rays(verts, c2w)
#         # rays_o, rays_d, viewdirs = get_rays(focal, c2w)
#         viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)

#         # Create ray batch
#         near = near.unsqueeze(-1) * torch.ones_like(rays_d[..., :1])
#         far = far.unsqueeze(-1) * torch.ones_like(rays_d[..., :1])
#         rays = torch.cat([rays_o, rays_d, near, far], -1)
#         rays = torch.cat([rays, viewdirs], -1)
#         rays = rays.float()
#         # rgb, features, sdf, mask, xyz, eikonal_term =
#         return {
#             **self.render_rays(rays,
#                                styles=styles,
#                                return_eikonal=return_eikonal,
#                                **kwargs),
#             'viewdirs':
#             # viewdirs.permute(0,3,1,2).reshape(viewdirs.shape[0], 3, -1) # b, 64, 64, 3 -> b, 3, 4096
#             viewdirs  # b, 64, 64, 3
#         }

# return rgb, features, sdf, mask, xyz, eikonal_term


# Generate mesh from xyz point cloud
def xyz2mesh(xyz):
    b, _, h, w = xyz.shape
    x, y = np.meshgrid(np.arange(h), np.arange(w))

    if not isinstance(xyz, torch.Tensor):
        xyz = torch.from_numpy(xyz)

    # Extract mesh faces from xyz maps
    tri = Delaunay(
        np.concatenate((x.reshape((h * w, 1)), y.reshape((h * w, 1))), 1))
    faces = tri.simplices

    # invert normals
    faces[:, [0, 1]] = faces[:, [1, 0]]

    # generate_meshes
    mesh = trimesh.Trimesh(
        xyz.squeeze(0).permute(1, 2, 0).reshape(h * w, 3).cpu().numpy(), faces)

    return mesh


################# Mesh rendering util functions #############################
def add_textures(meshes: Meshes, vertex_colors=None) -> Meshes:
    verts = meshes.verts_padded()
    if vertex_colors is None:
        vertex_colors = [torch.ones_like(verts[0])
                         ] * verts.shape[0]  # (N, V, 3)
    textures = TexturesVertex(verts_features=vertex_colors)
    meshes_t = Meshes(
        verts=verts,
        faces=meshes.faces_padded(),
        textures=textures,
        verts_normals=meshes.verts_normals_padded(),
    )
    return meshes_t


def create_mesh_renderer(
        cameras: FoVPerspectiveCameras,
        image_size: int = 256,
        blur_radius: float = 1e-6,
        light_location=((-0.5, 1., 5.0), ),
        device="cuda",
        **light_kwargs,
):
    """
    If don't want to show direct texture color without shading, set the light_kwargs as
    ambient_color=((1, 1, 1), ), diffuse_color=((0, 0, 0), ), specular_color=((0, 0, 0), )
    """
    # We will also create a Phong renderer. This is simpler and only needs to render one face per pixel.
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=blur_radius,
        faces_per_pixel=5,
    )
    # We can add a point light in front of the object.
    lights = PointLights(device=device,
                         location=light_location,
                         **light_kwargs)
    phong_renderer = MeshRenderer(rasterizer=MeshRasterizer(
        cameras=cameras, raster_settings=raster_settings),
                                  shader=SoftPhongShader(device=device,
                                                         cameras=cameras,
                                                         lights=lights))

    return phong_renderer


## custom renderer
class MeshRendererWithDepth(nn.Module):

    def __init__(self, rasterizer, shader):
        super().__init__()
        self.rasterizer = rasterizer
        self.shader = shader

    def forward(self, meshes_world, **kwargs) -> torch.Tensor:
        fragments = self.rasterizer(meshes_world, **kwargs)
        images = self.shader(fragments, meshes_world, **kwargs)
        return images, fragments.zbuf


def create_depth_mesh_renderer(
    cameras: FoVPerspectiveCameras,
    image_size: int = 256,
    blur_radius: float = 1e-6,
    device="cuda",
    **light_kwargs,
):
    """
    If don't want to show direct texture color without shading, set the light_kwargs as
    ambient_color=((1, 1, 1), ), diffuse_color=((0, 0, 0), ), specular_color=((0, 0, 0), )
    """
    # We will also create a Phong renderer. This is simpler and only needs to render one face per pixel.
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=blur_radius,
        faces_per_pixel=17,
    )
    # We can add a point light in front of the object.
    lights = PointLights(device=device,
                         location=((-0.5, 1., 5.0), ),
                         **light_kwargs)
    renderer = MeshRendererWithDepth(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings,
            # device=device,
        ),
        shader=SoftPhongShader(device=device, cameras=cameras, lights=lights))

    return renderer


"""
.. _moeller_ray_trace_example:

Visualize the Moeller–Trumbore Algorithm
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example demonstrates the Moeller–Trumbore intersection algorithm
using pyvista.

For additional details, please reference the following:

- `Möller–Trumbore intersection algorithm <https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm>`_
- `Fast Minimum Storage Ray Triangle Intersectio <https://cadxfem.org/inf/Fast%20MinimumStorage%20RayTriangle%20Intersection.pdf>`_

First, define the ray triangle intersection method.
"""


def ray_triangle_intersection(ray_start, ray_vec, triangle):
    """Moeller–Trumbore intersection algorithm.

    Parameters
    ----------
    ray_start : np.ndarray
        Length three numpy array representing start of point.

    ray_vec : np.ndarray
        Direction of the ray.

    triangle : np.ndarray
        ``3 x 3`` numpy array containing the three vertices of a
        triangle.

    Returns
    -------
    bool
        ``True`` when there is an intersection.

    tuple
        Length three tuple containing the distance ``t``, and the
        intersection in unit triangle ``u``, ``v`` coordinates.  When
        there is no intersection, these values will be:
        ``[np.nan, np.nan, np.nan]``

    """
    # define a null intersection
    null_inter = np.array([np.nan, np.nan, np.nan])

    # break down triangle into the individual points
    v1, v2, v3 = triangle
    eps = 0.000001

    # compute edges
    edge1 = v2 - v1
    edge2 = v3 - v1
    pvec = np.cross(ray_vec, edge2)
    det = edge1.dot(pvec)

    if abs(det) < eps:  # no intersection
        return False, null_inter
    inv_det = 1.0 / det
    tvec = ray_start - v1
    u = tvec.dot(pvec) * inv_det

    if u < 0.0 or u > 1.0:  # if not intersection
        return False, null_inter

    qvec = np.cross(tvec, edge1)
    v = ray_vec.dot(qvec) * inv_det
    if v < 0.0 or u + v > 1.0:  # if not intersection
        return False, null_inter

    t = edge2.dot(qvec) * inv_det
    if t < eps:
        return False, null_inter

    return True, np.array([t, u, v])
