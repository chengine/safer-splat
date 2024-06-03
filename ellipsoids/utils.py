import torch
import numpy as np
import open3d as o3d

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def rot_z(thetas, n_dim=2):
    if n_dim == 2:
        row1 = torch.stack([torch.cos(thetas), -torch.sin(thetas)], dim=-1)
        row2 = torch.stack([torch.sin(thetas), torch.cos(thetas)], dim=-1)

        return torch.stack([row1, row2], dim=1)
    elif n_dim == 3:
        row1 = torch.stack([torch.cos(thetas), -torch.sin(thetas), torch.zeros_like(thetas)], dim=-1)
        row2 = torch.stack([torch.sin(thetas), torch.cos(thetas), torch.zeros_like(thetas)], dim=-1)
        row3 = torch.stack([torch.zeros_like(thetas), torch.zeros_like(thetas), torch.ones_like(thetas)], dim=-1)
        
        return torch.stack([row1, row2, row3], dim=1)
    
def fibonacci_sphere(n=100):

    points = []
    phi = np.pi * (np.sqrt(5.) - 1.)  # golden angle in radians

    samples = torch.arange(n)
    y = 1. - (samples / (n-1)) * 2.  # y goes from 1 to -1
    radius = torch.sqrt(1 - y * y)  # radius at y

    theta = phi * samples  # golden angle increment

    x = torch.cos(theta) * radius
    z = torch.sin(theta) * radius

    points = torch.stack([x, y, z], axis=-1)

    return points

def fibonacci_ellipsoid(means, rotations, scalings, kappa, n=20):
    N, d = means.shape

    if d == 3:
        points = fibonacci_sphere(n).to(device)        # N x 3
    elif d == 2:
        t = torch.linspace(0, 2*np.pi, n, device=device)
        points = torch.stack([torch.cos(t), torch.sin(t)], dim=1)
    else:
        raise ValueError(f"Dimension {d} not supported")

    new_points = (scalings[:, None, :] + kappa) * points[None, :, :]      # B x N x 3

    new_points = torch.transpose(torch.bmm(rotations, torch.transpose(new_points, 1, 2)), 1, 2)
    new_points = new_points + means[:, None, :]

    return new_points

def create_gs_mesh(means, rotations, scalings, colors, res=4, transform=None, scale=None):
    scene = o3d.geometry.TriangleMesh()

    # Nerfstudio performs the transform first, then does the scaling
    if scale is not None:
        means = means * scale
        scalings = scalings * scale

    if transform is not None:
        rot = transform[:3, :3]
        t = transform[:3, -1]

        means = np.matmul(rot, means[..., None]).squeeze() + t

        rotations = np.matmul(rot, rotations)

    for i, (mean, R, S, col) in enumerate(zip(means, rotations, scalings, colors)):
        one_gs_mesh = o3d.geometry.TriangleMesh.create_sphere(resolution=res)
        points = np.asarray(one_gs_mesh.vertices)
        new_points = points * S[None]
        one_gs_mesh.vertices = o3d.utility.Vector3dVector(new_points)
        one_gs_mesh = one_gs_mesh.paint_uniform_color(col)
        one_gs_mesh = one_gs_mesh.rotate(R)
        one_gs_mesh = one_gs_mesh.translate(mean)
        scene += one_gs_mesh

    return scene