#%%
from nerfstudio.utils.eval_utils import eval_setup
from ellipsoids.mesh_utils import create_gs_mesh
from ellipsoids.covariance_utils import quaternion_to_rotation_matrix
from pathlib import Path
import torch
import json
import os
import open3d as o3d
import json
from scipy.stats import chi2
from ns_utils.nerfstudio_utils import GaussianSplat
import imageio
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def convert_sh_to_rgb(sh):
    C0 = 0.28209479177387814
    rgbs =  sh * C0
    return torch.clamp(rgbs + 0.5, 0.0, 1.0)

# %%

# path_to_gsplat = Path('outputs/stonehenge/splatfacto/2024-09-11_100724/config.yml')
# path_to_gsplat = Path('outputs/statues/splatfacto/2024-09-11_095852/config.yml')
path_to_gsplat = Path('outputs/flight/splatfacto/2024-09-12_172434/config.yml')
gs_pipeline = GaussianSplat(path_to_gsplat, dataset_mode='train')

poses = gs_pipeline.get_poses()

# if os.path.isfile(path_to_gsplat):
#     with open('outputs/colmap_in_mocap/splatfacto/2024-04-15_192132/dataparser_transforms.json', 'r') as f:
#         meta = json.load(f)

#     scale = 1./ meta["scale"]
#     transform = np.eye(4)
#     transform[:3] = np.array(meta["transform"])
#     transform = np.linalg.inv(transform)

#%%
gs_params = gs_pipeline.pipeline.model.get_gaussian_param_groups()
# %%
means = gs_params['means'][0]
sh = gs_params['features_dc'][0]     # 0th order spherical harmonic coefficient
colors = convert_sh_to_rgb(sh)
opacity = torch.sigmoid(gs_params['opacities'][0])
scalings = torch.exp(gs_params['scales'][0])
rotation = gs_params['quats'][0]

# for i, pose in enumerate(poses):
#     output = gs_pipeline.render(pose)

#     imageio.imwrite(f'renders/mocap/render_{i}.png', (255*output['rgb'].detach().cpu().numpy()).astype(np.uint8))
#     np.save( f'renders/mocap/depth_{i}.npy', output['depth'].detach().cpu().numpy() )

#%%
H, W, K = gs_pipeline.get_camera_intrinsics()
with open('poses.json', 'w') as f:
    data = {
        'pose0': poses[10].tolist(),
        'pose1': poses[11].tolist(),
        'H': H.item(),
        'W': W.item(),
        'K': K.tolist()
    }
    json.dump(data, f, indent=4)

#%% Render from mocap

# new_poses = np.stack([np.eye(4)]*len(poses), axis=0)
# new_poses[:, :3, :3] = np.matmul(transform[:3, :3], poses[:, :3, :3])
# new_poses[:, :3, -1] = np.matmul(transform[:3, :3], scale*poses[:, :3, -1][..., None]).squeeze() + transform[:3, -1]

# data = {
#     'transform_matrix': [pose.tolist() for pose in new_poses]
# }

# with open('renders/transforms.json', 'w') as f:
#     json.dump(data, f, indent=4)

#%% Segment out only supervised tiles

# mask = (num_tiles_hit > 1000)

# means = means[mask]
# colors = colors[mask]
# opacity = opacity[mask]
# scalings = scalings[mask]
# rotation = rotation[mask]

#%% 
# covs = compute_cov(rotation, scalings)

data = {
    'means': (torch.matmul(torch.tensor(transform[:3, :3], dtype=torch.float32).cuda(), scale*means.unsqueeze(-1)).squeeze() + torch.tensor(transform[:3, -1], dtype=torch.float32).cuda()).tolist(),
    'colors': colors.tolist(),
    'opacities': opacity.tolist(),
    'scalings': (scale*scalings).tolist(),
    'rotations': torch.matmul(torch.tensor(transform[:3, :3], dtype=torch.float32).cuda(), quaternion_to_rotation_matrix(rotation)[:, :3, :3]).tolist()
}

with open('gaussians.json', 'w') as f:
    json.dump(data, f, indent=4)

# %%

opac_mask = (opacity >= 0.).squeeze()

cutoff = 1.# 3.36 # np.sqrt(chi2.ppf(0.99, 3))
# culling = torch.sqrt(-2*torch.log(1e-2 / opacity))

prob_scalings = cutoff*scalings
# prob_scalings = cutoff*(opacity**(1/3))*scalings[opac_mask]

# rectify_mask = (culling.squeeze() <= cutoff)

# prob_scalings = scalings[opac_mask]
# prob_scalings[rectify_mask] = culling[rectify_mask]*scalings[rectify_mask]
# prob_scalings[~rectify_mask] = cutoff*scalings[~rectify_mask]

# prob_scalings = cutoff*scalings[opac_mask]
#%%

scene = create_gs_mesh(means[opac_mask].detach().cpu().numpy(), quaternion_to_rotation_matrix(rotation[opac_mask])[..., :3, :3].detach().cpu().numpy(), prob_scalings.detach().cpu().numpy(), colors[opac_mask].detach().cpu().numpy(), transform=None, scale=None, res=6)

# bound = np.array([
#     [-10., 10.],
#     [-5., 5.],
#     [-0.25, 2.5]
# ])

# BB = o3d.geometry.AxisAlignedBoundingBox()
# BB.max_bound = bound[:, -1]
# BB.min_bound = bound[:, 0]

# scene = scene.crop(BB)

# o3d.visualization.draw_geometries([scene])

#%%
o3d.io.write_triangle_mesh('scene.obj', scene, compressed=True, write_vertex_colors=True, write_vertex_normals=True, write_triangle_uvs=True, print_progress=True)
# %%
