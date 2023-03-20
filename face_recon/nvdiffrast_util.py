"""This script is the differentiable renderer for Deep3DFaceRecon_pytorch
    Attention, anti-aliasing step is missing in current version.
"""

from typing import List

import numpy as np
import torch

from torch import nn

import nvdiffrast.torch as dr


def ndc_projection(x=0.1, n=1.0, f=50.0):
    return np.array(
        [[n / x, 0, 0, 0], [0, n / -x, 0, 0], [0, 0, -(f + n) / (f - n), -(2 * f * n) / (f - n)], [0, 0, -1, 0]]
    ).astype(np.float32)


class MeshRenderer(nn.Module):
    def __init__(self, rasterize_fov, znear=0.1, zfar=10, rasterize_size=224, device="cuda"):
        super().__init__()
        self.device = device
        self.rasterize_size = rasterize_size
        self.r_size = int(rasterize_size)

        x = np.tan(np.deg2rad(rasterize_fov * 0.5)) * znear
        ndc_projection_tensor = torch.tensor(ndc_projection(x=x, n=znear, f=zfar)).to(device)
        one_tensor = torch.tensor([1.0, -1, -1, 1]).to(device)
        self.ndc_proj = ndc_projection_tensor.matmul(torch.diag(one_tensor))

        self.glctx = dr.RasterizeGLContext(device=device)

    def forward(self, vertex, tri, feat=None):
        """
        Return:
            mask               -- torch.tensor, size (B, 1, H, W)
            depth              -- torch.tensor, size (B, 1, H, W)
            features(optional) -- torch.tensor, size (B, C, H, W) if feat is not None

        Parameters:
            vertex          -- torch.tensor, size (B, N, 3)
            tri             -- torch.tensor, size (B, M, 3) or (M, 3), triangles
            feat(optional)  -- torch.tensor, size (B, C), features
        """
        # trans to homogeneous coordinates of 3d vertices, the direction of y is the same as v
        if vertex.shape[-1] == 3:
            vertex = torch.cat([vertex, torch.ones([*vertex.shape[:2], 1]).to(self.device)], dim=-1)
            vertex[..., 1] = -vertex[..., 1]

        vertex_ndc = vertex @ self.ndc_proj.t()

        ranges = None
        if isinstance(tri, List) or len(tri.shape) == 3:
            vum = vertex_ndc.shape[1]
            fnum = torch.tensor([f.shape[0] for f in tri]).unsqueeze(1).to(self.device)
            fstartidx = torch.cumsum(fnum, dim=0) - fnum
            ranges = torch.cat([fstartidx, fnum], axis=1).type(torch.int32).cpu()  # must be on cpu
            for i in range(tri.shape[0]):
                tri[i] = tri[i] + i * vum
            vertex_ndc = torch.cat(vertex_ndc, dim=0)
            tri = torch.cat(tri, dim=0)

        # for range_mode vetex: [B*N, 4], tri: [B*M, 3], for instance_mode vetex: [B, N, 4], tri: [M, 3]
        tri = tri.type(torch.int32).contiguous()
        rast_out, _ = dr.rasterize(
            self.glctx, vertex_ndc.contiguous(), tri, resolution=[self.r_size, self.r_size], ranges=ranges
        )

        depth, _ = dr.interpolate(vertex.reshape([-1, 4])[..., 2].unsqueeze(1).contiguous(), rast_out, tri)
        depth = depth.permute(0, 3, 1, 2)
        mask = (rast_out[..., 3] > 0).float().unsqueeze(1)
        depth = mask * depth

        image = None  # image can be flow, when given two face project diff
        if feat is not None:
            image, _ = dr.interpolate(feat, rast_out, tri)
            image = image.permute(0, 3, 1, 2)
            image = mask * image

        return mask, depth, image

    def compute_mask_flow(self, vertex, tri, feat):
        """
        Return:
            mask               -- torch.tensor, size (B, 1, H, W)
            flow               -- torch.tensor, size (B, 2, H, W)

        Parameters:
            vertex          -- torch.tensor, size (B, N, 3)
            tri             -- torch.tensor, size (B, M, 3) or (M, 3), triangles
            feat            -- torch.tensor, size (B, 2), features
        """
        # trans to homogeneous coordinates of 3d vertices, the direction of y is the same as v
        if vertex.shape[-1] == 3:
            vertex = torch.cat([vertex, torch.ones([*vertex.shape[:2], 1]).to(self.device)], dim=-1)
            vertex[..., 1] = -vertex[..., 1]

        vertex_ndc = vertex @ self.ndc_proj.t()

        ranges = None
        if isinstance(tri, List) or len(tri.shape) == 3:
            vum = vertex_ndc.shape[1]
            fnum = torch.tensor([f.shape[0] for f in tri]).unsqueeze(1).to(self.device)
            fstartidx = torch.cumsum(fnum, dim=0) - fnum
            ranges = torch.cat([fstartidx, fnum], axis=1).type(torch.int32).cpu()  # must be on cpu
            for i in range(tri.shape[0]):
                tri[i] = tri[i] + i * vum
            vertex_ndc = torch.cat(vertex_ndc, dim=0)
            tri = torch.cat(tri, dim=0)

        # for range_mode vetex: [B*N, 4], tri: [B*M, 3], for instance_mode vetex: [B, N, 4], tri: [M, 3]
        tri = tri.type(torch.int32).contiguous()
        rast_out, _ = dr.rasterize(
            self.glctx, vertex_ndc.contiguous(), tri, resolution=[self.r_size, self.r_size], ranges=ranges
        )

        mask = (rast_out[..., 3] > 0).float().unsqueeze(1)

        flow = None  # flow can be flow, when given two face project diff
        if feat is not None:
            flow, _ = dr.interpolate(feat, rast_out, tri)
            flow = flow.permute(0, 3, 1, 2)
            flow = mask * flow

        return mask, flow
