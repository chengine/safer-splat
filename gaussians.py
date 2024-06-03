#%%
from nerfstudio.utils.eval_utils import eval_setup
from gs_utils.gs_utils import *
from viz_utils.viz_utils import *
from pathlib import Path
import torch
import json
import os
import open3d as o3d
import json
from scipy.stats import chi2
from pose_estimator.utils.nerf_utils import NeRF
import imageio

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# class NeRFWrapper():
#     def __init__(self, config_fp) -> None:

#         config_path = Path(config_fp + "/config.yml") # Path to config file 

#         if os.path.isfile(config_fp + "/dataparser_transforms.json"):
#             with open(config_fp + "/dataparser_transforms.json", 'r') as f:
#                 meta = json.load(f)

#             self.scale = meta["scale"]
#             transform = meta["transform"]
#             self.transform = torch.tensor(transform, device=device, dtype=torch.float32)

#         else: 
#             self.scale = 1.
#             self.transform = torch.eye(4).to(device)

#         # This is from world coordinates to scene coordinates
#         # rot = R.from_euler('xyz', [0, 0., 30], degrees=True)
#         # world_transform = np.eye(4)
#         # world_transform[:3, :3] = rot.as_matrix()
#         # world_transform[:3, -1] = np.array([5., 0., 0.75])
#         # world_transform = np.linalg.inv(world_transform)
#         # world_transform = np.eye(4, dtype=np.float32)

#         # Prepare model
#         _, self.pipeline, _, _ = eval_setup(
#             config_path, 
#             test_mode="inference",
#         )

#     def data_frame_to_ns_frame(self, points):
#         transformed_points = (self.transform[:3, :3]@points.T).T + self.transform[:3, -1][None,:]
#         transformed_points *= self.scale

#         return transformed_points

#     def ns_frame_to_data_frame(self, points):
#         transformed_points = points / self.scale
#         transformed_points = transformed_points - self.transform[:3, -1][None, :]
#         transformed_points = (self.transform[:3, :3].T @ transformed_points.T).T

#         return transformed_points

def convert_sh_to_rgb(sh):
    C0 = 0.28209479177387814
    rgbs =  sh * C0
    return torch.clamp(rgbs + 0.5, 0.0, 1.0)

# %%
# gs_pipeline = NeRFWrapper('outputs/flightroom/splatfacto/2024-04-04_121448')
# gs_pipeline = NeRFWrapper('outputs/splat/splatfacto/2024-04-04_154009')

# gs_pipeline = NeRF(Path('outputs/splat/splatfacto/2024-04-04_154009/config.yml'))

# gs_pipeline = NeRF(Path('outputs/colmap_in_mocap/splatfacto/2024-04-07_203937/config.yml'), dataset_mode='train')

gs_pipeline = NeRF(Path('outputs/colmap_in_mocap/splatfacto/2024-04-15_192132/config.yml'), dataset_mode='train')

poses = gs_pipeline.get_poses()

if os.path.isfile('outputs/colmap_in_mocap/splatfacto/2024-04-15_192132/dataparser_transforms.json'):
    with open('outputs/colmap_in_mocap/splatfacto/2024-04-15_192132/dataparser_transforms.json', 'r') as f:
        meta = json.load(f)

    scale = 1./ meta["scale"]
    transform = np.eye(4)
    transform[:3] = np.array(meta["transform"])
    transform = np.linalg.inv(transform)

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

cutoff = 3.36 # np.sqrt(chi2.ppf(0.99, 3))
# culling = torch.sqrt(-2*torch.log(1e-2 / opacity))

prob_scalings = cutoff*scalings
# prob_scalings = cutoff*(opacity**(1/3))*scalings[opac_mask]

# rectify_mask = (culling.squeeze() <= cutoff)

# prob_scalings = scalings[opac_mask]
# prob_scalings[rectify_mask] = culling[rectify_mask]*scalings[rectify_mask]
# prob_scalings[~rectify_mask] = cutoff*scalings[~rectify_mask]

# prob_scalings = cutoff*scalings[opac_mask]
#%%

scene = create_gs_mesh(means[opac_mask].detach().cpu().numpy(), quaternion_to_rotation_matrix(rotation[opac_mask])[..., :3, :3].detach().cpu().numpy(), prob_scalings.detach().cpu().numpy(), colors[opac_mask].detach().cpu().numpy(), transform=transform, scale=scale, res=6)

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
