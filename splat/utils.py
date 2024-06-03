import json
import torch
import numpy as np
import open3d as o3d
from downsampling.utils import mvee, mvee_batched, outer_ellipsoid
from ellipsoids.utils import fibonacci_ellipsoid, rot_z, create_gs_mesh
from ellipsoids.plots import plot_ellipse
from ellipsoids.gs_utils import quaternion_to_rotation_matrix
import open3d as o3d
from scipy.spatial import KDTree
from ellipsoids.gs_utils import compute_cov

def batch_mahalanobis_distance(x, means, covs):
    # Computes the Mahalanobis distance of a batch of points x to a batch of Gaussians with means and covariances.
    # x: n
    # means: B x n
    # covs: B x n x n

    # n = x.shape
    # B, n, n = covs.shape

    x = x.unsqueeze(0)      # 1 x n

    diff = x - means
    mahalanobis = torch.einsum('bm,bmn,bn->b', diff, covs, diff)
    grad = 2 * torch.bmm(covs, diff[..., None]).squeeze()

    return mahalanobis, grad

class GSplat():
    def __init__(self, filepath, device, kdtree=False):
        self.device = device

        self.load_gsplat(filepath)

        if kdtree:
            self.kdtree = KDTree(self.means.cpu().numpy())

    def load_gsplat(self, filepath, normalized=False):
        with open(filepath, 'r') as f:
            data = json.load(f)

        self.means = torch.tensor(data['means']).to(dtype=torch.float32, device=self.device)
        self.rots = torch.tensor(data['rotations']).to(dtype=torch.float32, device=self.device)
        self.colors = torch.tensor(data['colors']).to(dtype=torch.float32, device=self.device)

        if normalized:
            self.opacities = torch.tensor(data['opacities']).to(dtype=torch.float32, device=self.device)
            self.scales = torch.tensor(data['scalings']).to(dtype=torch.float32, device=self.device)
        else:
            self.opacities = torch.sigmoid(torch.tensor(data['opacities']).to(dtype=torch.float32, device=self.device))
            self.scales = torch.exp(torch.tensor(data['scalings']).to(dtype=torch.float32, device=self.device))

        # Computes Sigma inverse
        self.cov_inv = compute_cov(self.rots, 1/self.scales)

    def save_mesh(self, filepath):
        scene = create_gs_mesh(self.means.cpu().numpy(), quaternion_to_rotation_matrix(self.rots).cpu().numpy(), self.scales.cpu().numpy(), self.colors.cpu().numpy(), res=4, transform=None, scale=None)
        o3d.io.write_triangle_mesh(filepath, scene)

    def query_distance(self, x, radius=None):
        # Queries the Mahalanobis distance of x to the GSplat. If radius is provided, returns the indices of the GSplats within the radius using a KDTree.

        if radius is None:
            return batch_mahalanobis_distance(x, self.means, self.cov_inv)
            
        else:
            assert radius is not None, 'Radius must be provided for KDTree query.'
            idx = self.kdtree.query_ball_point(x.cpu().numpy(), radius)

            return batch_mahalanobis_distance(x, self.means[idx], self.cov_inv[idx])

