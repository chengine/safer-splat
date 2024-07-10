import json
import torch
import numpy as np
# import open3d as o3d
from downsampling.utils import mvee, mvee_batched, outer_ellipsoid
from ellipsoids.utils import fibonacci_ellipsoid, rot_z, create_gs_mesh
from ellipsoids.plots import plot_ellipse
from ellipsoids.gs_utils import quaternion_to_rotation_matrix
import open3d as o3d
from scipy.spatial import KDTree
from ellipsoids.gs_utils import compute_cov
from tqdm import tqdm
import time


def batch_mahalanobis_distance(x, means, covs):
    # Computes the Mahalanobis distance of a batch of points x to a batch of Gaussians with means and covariances.
    # x: n
    # means: B x n
    # covs: B x n x n

    # n = x.shape
    # B, n, n = covs.shape

    eps = 1e-6

    x = x.unsqueeze(0)      # 1 x n

    # we only want x y z 
    x = x[:, :3]

    diff = x - means
    # print(diff)
    mahalanobis = torch.einsum('bm,bmn,bn->b', diff, covs, diff)


    # Compute the eigenvalues for each covariance matrix
    eigenvalues = torch.linalg.eigvals(covs)
    
    # Get the maximum eigenvalue for each covariance matrix
    max_eigenvalues = torch.min(eigenvalues.real, dim=-1)[0] # Use the real part

    # print(f"covs shape: {covs}")

    det_covs = torch.det(covs)
    # sdf = mahalanobis/ torch.sqrt(det_covs) - 1
    sdf = torch.sqrt(mahalanobis) - 1

    # normalize by the determinant of the covariance matrix
    sdf = mahalanobis / (max_eigenvalues + eps)

    grad = 2 * torch.bmm(covs, diff[..., None]).squeeze() 

    # also normlize the gradient
    grad = grad / (max_eigenvalues[..., None] + eps)

    hessian = 2 * covs
    # normalize the hessian
    hessian = hessian / (max_eigenvalues[..., None, None] + eps)

    return sdf, grad, hessian


def batch_euclidean_distance(x, means, covs):
    # Computes the Euclidean distance of a batch of points x to a batch of Gaussians with means and covariances.
    # Assumes each Gaussian is a sphere with a radius of the largest eigenvalue of the covariance matrix.
    # x: n
    # means: B x n
    # covs: B x n x n

    eps = 1e-6

    x = x.unsqueeze(0)  # 1 x n

    # we only want x y z 
    x = x[:, :3]

    diff = x - means  # B x 3

    # Compute the Euclidean distance
    euclidean_distance = torch.norm(diff, dim=1)

    # Compute the eigenvalues for each covariance matrix
    eigenvalues = torch.linalg.eigvals(covs)
    
    # Get the maximum eigenvalue for each covariance matrix
    max_eigenvalues = torch.max(eigenvalues.real, dim=-1)[0]  # Use the real part

    # Normalize the Euclidean distance by the largest eigenvalue
    normalized_distance = euclidean_distance -.025 #/ (max_eigenvalues + eps)

    # Compute the gradient
    grad = diff / (euclidean_distance.unsqueeze(1) + eps)  # B x 3

    # Normalize the gradient by the largest eigenvalue
    grad = grad #/ (max_eigenvalues.unsqueeze(1) + eps)

    # Compute the Hessian in a batched manner
    B, n = diff.shape
    I = torch.eye(n, device=x.device).unsqueeze(0).expand(B, -1, -1)  # B x 3 x 3
    d = diff.unsqueeze(2)  # B x 3 x 1
    dT = diff.unsqueeze(1)  # B x 1 x 3
    euclidean_distance_squared = euclidean_distance ** 2 + eps
    euclidean_distance_cubed = euclidean_distance + eps

    hessian = (I - (d @ dT) / euclidean_distance_squared.unsqueeze(1).unsqueeze(2)) / euclidean_distance_cubed.unsqueeze(1).unsqueeze(2)
    
    
    # Normalize the Hessian by the largest eigenvalue
    hessian = hessian #/ (max_eigenvalues.unsqueeze(1).unsqueeze(1) + eps)

    return normalized_distance, grad, hessian


class GSplat():
    def __init__(self, filepath, device, kdtree=False):
        self.device = device

        self.load_gsplat(filepath)

        if kdtree:
            self.kdtree = KDTree(self.means.cpu().numpy())

    # def load_gsplat(self, filepath, normalized=False):
    #     with open(filepath, 'r') as f:
    #         data = json.load(f)

    #     self.means = torch.tensor(data['means']).to(dtype=torch.float32, device=self.device)
    #     self.rots = torch.tensor(data['rotations']).to(dtype=torch.float32, device=self.device)
    #     self.colors = torch.tensor(data['colors']).to(dtype=torch.float32, device=self.device)

    #     if normalized:
    #         self.opacities = torch.tensor(data['opacities']).to(dtype=torch.float32, device=self.device)
    #         self.scales = torch.tensor(data['scalings']).to(dtype=torch.float32, device=self.device)
    #     else:
    #         self.opacities = torch.sigmoid(torch.tensor(data['opacities']).to(dtype=torch.float32, device=self.device))
    #         self.scales = torch.exp(torch.tensor(data['scalings']).to(dtype=torch.float32, device=self.device))

    #     # Computes Sigma inverse
    #     self.cov_inv = compute_cov(self.rots, 1/self.scales)
            
    def load_gsplat(self, filepath, normalized=False):
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        keys = ['means', 'rotations', 'colors', 'opacities', 'scalings']
        tensors = {}

        # Measure time for loading tensors
        start_time = time.time()
        for key in keys:
            tensors[key] = torch.tensor(data[key]).to(dtype=torch.float32, device=self.device)
        print(f"Loading tensors took {time.time() - start_time:.4f} seconds")
        
        # Measure time for setting attributes
        start_time = time.time()
        self.means = tensors['means']
        self.rots = tensors['rotations']
        self.colors = tensors['colors']
        self.opacities = tensors['opacities']
        self.scales = tensors['scalings']
        print(f"Setting attributes took {time.time() - start_time:.4f} seconds")

        # Print tensor sizes
        print(f"Opacities tensor size: {self.opacities.size()}")
        print(f"Scales tensor size: {self.scales.size()}")

        # Measure time for normalization
        if not normalized:
            start_time = time.time()
            self.opacities = torch.sigmoid(self.opacities)
            self.scales = torch.exp(self.scales)
            print(f"Normalization took {time.time() - start_time:.4f} seconds")

        # Measure time for computing Sigma inverse
        start_time = time.time()
        self.cov_inv = compute_cov(self.rots, 1 / self.scales)
        print(f"Computing Sigma inverse took {time.time() - start_time:.4f} seconds")

    def save_mesh(self, filepath):
        scene = create_gs_mesh(self.means.cpu().numpy(), quaternion_to_rotation_matrix(self.rots).cpu().numpy(), self.scales.cpu().numpy(), self.colors.cpu().numpy(), res=4, transform=None, scale=None)
        o3d.io.write_triangle_mesh(filepath, scene)

    def query_distance(self, x, radius=None):
        # Queries the Mahalanobis distance of x to the GSplat. If radius is provided, returns the indices of the GSplats within the radius using a KDTree.

        if radius is None:
            return batch_euclidean_distance(x, self.means, self.cov_inv)
            
        else:
            assert radius is not None, 'Radius must be provided for KDTree query.'
            idx = self.kdtree.query_ball_point(x.cpu().numpy(), radius)

            return batch_euclidean_distance(x, self.means[idx], self.cov_inv[idx])

