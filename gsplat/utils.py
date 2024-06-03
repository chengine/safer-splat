import json
import torch
import numpy as np
import open3d as o3d
from downsampling.utils import mvee, mvee_batched, outer_ellipsoid
from ellipsoids.utils import fibonacci_ellipsoid, rot_z, create_gs_mesh
from ellipsoids.plots import plot_ellipse
from ellipsoids.gs_utils import quaternion_to_rotation_matrix
import open3d as o3d

class GSplat():
    def __init__(self, filepath, device):
        self.device = device

        self.load_gsplat(filepath)

    def load_gsplat(self, filepath, normalized=False):
        with open(filepath, 'r') as f:
            data = json.load(f)

        self.means = torch.tensor(data['means']).astype(torch.float32, device=self.device)
        self.rots = torch.tensor(data['rotations']).to(dtype=torch.float32, device=self.device)
        self.colors = torch.tensor(data['colors']).to(dtype=torch.float32, device=self.device)

        if normalized:
            self.opacities = torch.tensor(data['opacities']).to(dtype=torch.float32, device=self.device)
            self.scales = torch.tensor(data['scalings']).to(dtype=torch.float32, device=self.device)
        else:
            self.opacities = torch.sigmoid(torch.tensor(data['opacities']).to(dtype=torch.float32, device=self.device))
            self.scales = torch.exp(torch.tensor(data['scalings']).to(dtype=torch.float32, device=self.device))

    def save_mesh(self, filepath):
        scene = create_gs_mesh(self.means.cpu().numpy(), quaternion_to_rotation_matrix(self.rots).cpu().numpy(), self.scales.cpu().numpy(), self.colors.cpu().numpy(), res=4, transform=None, scale=None)
        o3d.io.write_triangle_mesh(filepath, scene)

    

