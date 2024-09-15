#%%
import numpy as np 
import torch
import time
import open3d as o3d 
from splat.gsplat_utils import GSplatLoader
from ellipsoids.mesh_utils import create_gs_mesh
from ellipsoids.covariance_utils import quaternion_to_rotation_matrix
from gsplat.cuda._wrapper import rasterize_to_indices_in_range
from pathlib import Path
from matplotlib import pyplot as plt
import cv2

def SH2RGB(sh):
    """
    Converts from the 0th spherical harmonic coefficient to RGB values [0,1]
    """
    C0 = 0.28209479177387814
    return sh * C0 + 0.5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# config_path = Path('outputs/splato/splatfacto/2024-08-04_235629/config.yml')
config_path = Path('outputs/old_union2/splatfacto/2024-09-02_151414/config.yml')

loader = GSplatLoader(config_path, device=device)
splat = loader.splat
#%%
# Recovers dataset poses
poses = splat.get_poses()

i = 25
# Renders splat from a pose. We call this to get the intermediate variables from the rasterization function.
tnow = time.time()
output = splat.render(poses[i])
print("Elapsed: ", time.time() - tnow)

og_image = output['rgb'].cpu().numpy()
depth_image = output['depth'].cpu().numpy()
depth_image = (depth_image - depth_image.min()) / (2 - depth_image.min())

cm = plt.get_cmap('jet')
depth_image = cm(depth_image.squeeze())[:, :, :3].astype(np.float32)

cv2.imwrite(f'rgb_{i}.png', cv2.cvtColor( og_image * 255, cv2.COLOR_BGR2RGB))
cv2.imwrite(f'depth_{i}.png', cv2.cvtColor( depth_image * 255, cv2.COLOR_BGR2RGB))

info = output["info"]
# pixel_ids/gaussian ids are sorted in order of depth of gaussians
gs_ids, pixel_ids, camera_ids = rasterize_to_indices_in_range(
    0,
    1000,
    torch.ones(1, info["height"], info["width"], device=device),
    info["means2d"],
    info["conics"],
    info["opacities"],
    info["width"],
    info["height"],
    info["tile_size"],
    info["isect_offsets"],
    info["flatten_ids"],
)

# cv2.imwrite(f'ellipsoids_{i}.png', cv2.cvtColor( og_image * 255, cv2.COLOR_BGR2RGB))

gs_params = splat.pipeline.model.get_gaussian_param_groups()

ids_to_choose = torch.unique(gs_ids)

means = gs_params['means'][0][ids_to_choose]
sh = gs_params['features_dc'][0][ids_to_choose]     # 0th order spherical harmonic coefficient
colors = SH2RGB(sh)
opacity = torch.sigmoid(gs_params['opacities'][0][ids_to_choose])
scaling = torch.exp(gs_params['scales'][0][ids_to_choose])
rotation = gs_params['quats'][0][ids_to_choose]
prob_scalings = scaling.squeeze()

#%%
# render_colors = 0.5*np.ones_like(colors.detach().cpu().numpy())
scene = create_gs_mesh(means.detach().cpu().numpy(), quaternion_to_rotation_matrix(rotation)[..., :3, :3].detach().cpu().numpy(), prob_scalings.detach().cpu().numpy(), colors.detach().cpu().numpy(), res=4)
# o3d.visualization.draw_geometries([scene])
# o3d.io.write_triangle_mesh('ellipsoids.obj', scene, print_progress=True)

#%%
import json
from PIL import ImageColor

cm = plt.get_cmap('turbo')

pose_to_render = np.eye(4)
pose_to_render[:3] = poses[i].detach().cpu().numpy()

opengl_to_opencv = np.array([ 

    [1., 0., 0., 0.],

    [0., -1., 0., 0.],

    [0., 0., -1., 0.],

    [0., 0., 0., 1.]

])

def construct_body_to_world_transform(current_point, next_point):
    direction = next_point - current_point
    direction /= np.linalg.norm(direction)
    up = np.array([0, 0, 1])  # Assuming z-axis is up
    right = np.cross(direction, up)
    right /= np.linalg.norm(right)
    up = np.cross(right, direction)
    rotation_matrix = np.column_stack((right, direction, up))
    return rotation_matrix

pose_to_render = pose_to_render @ opengl_to_opencv
pose_to_render = np.linalg.inv(pose_to_render)

H, W, K = splat.get_camera_intrinsics()

drones = []
for k in range(50):
    drone = o3d.io.read_triangle_mesh('geometries/quadcopter_drone/drone.obj')
    drone = drone.scale(0.0225, center=drone.get_center())

    drone_col = np.array(ImageColor.getcolor("#FBBc05", "RGB"))/255.

    # drone.paint_uniform_color(drone_col)
    drone = drone.compute_vertex_normals()

    rot_x_90 = np.array([
        [1., 0., 0., 0.],
        [0., 0., -1., 0.],
        [0., 1., 0., 0.],
        [0., 0., 0., 1.]
    ])

    drone = drone.transform(rot_x_90)
    drones.append(drone)

with open(f'trajs/old_union_ball-to-ellipsoid.json', 'r') as f:
    meta = json.load(f)

total_data = meta['total_data'][55:58]

drone_meta = []

for j, data in enumerate(total_data):
    traj = np.array(data['traj'])
    h_min = np.array(data['safety'])

    total_length = np.linalg.norm(traj[:-1] - traj[1:], axis=1).sum()
    chunk_length = total_length / 5

    equal_spaced_pts = []
    next_pts = []

    current_pt = traj[0, :3]
    for m, pt in enumerate(traj[:, :3]):
        if np.linalg.norm(pt - current_pt) > chunk_length:
            equal_spaced_pts.append(pt)
            current_pt = pt
            next_pts.append(traj[m+1, :3])

    equal_spaced_pts.append(traj[-1, :3])
    equal_spaced_pts = np.array(equal_spaced_pts)

    hmin_normalized = 1. - (h_min - h_min.min()) / (h_min.max() - h_min.min())

    point_traj_colors = cm(hmin_normalized)[:, :3]
    point_traj_colors = np.concatenate([point_traj_colors, point_traj_colors[-1][None]], axis=0)
    point_traj = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.array(data['traj'])[:, :3]))
    point_traj.colors = o3d.utility.Vector3dVector(point_traj_colors)

    drone_meta.append((equal_spaced_pts, next_pts, point_traj))

viz_drones = []

counter = 0
for m, meta_ in enumerate(drone_meta):
    equal_spaced_pts, next_pts, point_traj = meta_

    for now_pt, next_pt in zip(equal_spaced_pts, next_pts):
        drones[counter].translate(now_pt, relative=False)
        Rcw = construct_body_to_world_transform(now_pt, next_pt)
        drones[counter].rotate(Rcw, center=drones[counter].get_center())
        viz_drones.append(drones[counter])

        counter += 1


def custom_draw_geometry_with_custom_fov(pcd):

    vis = o3d.visualization.Visualizer()

    vis.create_window(width=W.item(), height=H.item())

    vis.add_geometry(pcd)

    # axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    # vis.add_geometry(axes)

    # for meta_ in drone_meta:
    #     equal_spaced_pts, next_pts, point_traj = meta_
    #     vis.add_geometry(point_traj)

    # for drone_ in viz_drones:
    #     vis.add_geometry(drone_)

    ctr = vis.get_view_control()

    intr = o3d.camera.PinholeCameraIntrinsic(W.item(), H.item(), K.cpu().numpy())
    params = o3d.camera.PinholeCameraParameters()
    params.intrinsic = intr
    params.extrinsic = pose_to_render

    # smaller_frustum_H = int(H.item() * 0.25)
    # smaller_frustum_W = int(W.item() * 0.25)
    # smaller_frustum_K = K.cpu().numpy() * 0.25

    # intr_cam = o3d.camera.PinholeCameraIntrinsic(smaller_frustum_W, smaller_frustum_H, smaller_frustum_K)
    # frustum = o3d.geometry.LineSet.create_camera_visualization(intr_cam, pose_to_render)
    # frustum.paint_uniform_color([0, 0, 0])
    # vis.add_geometry(frustum)

    asdf = ctr.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)

    if asdf:
        vis.run()

        image = vis.capture_screen_float_buffer()
        image = np.asarray(image)
        cv2.imwrite(f'viz/ellipsoids_{i}_viz_{j}.png', cv2.cvtColor( image * 255, cv2.COLOR_BGR2RGB))
        vis.destroy_window()
    else:
        print("Failed to set camera parameters")

custom_draw_geometry_with_custom_fov(scene)

#%%