#%%
import numpy as np 
import torch
import time
import open3d as o3d 
from splat.utils import *
from splat.spatial_utils import *
from splat.gs_utils import *
from gsplat.cuda._wrapper import rasterize_to_indices_in_range

def SH2RGB(sh):
    """
    Converts from the 0th spherical harmonic coefficient to RGB values [0,1]
    """
    C0 = 0.28209479177387814
    return sh * C0 + 0.5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# config_path = Path('outputs/splato/splatfacto/2024-08-04_235629/config.yml')
config_path = Path('outputs/old_union2/splatfacto/2024-09-02_151414/config.yml')

splat = NeRF(config_path, dataset_mode='train')

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
prob_scalings = 2*scaling.squeeze()

#%%

# render_colors = 0.5*np.ones_like(colors.detach().cpu().numpy())
scene = create_gs_mesh(means.detach().cpu().numpy(), quaternion_to_rotation_matrix(rotation)[..., :3, :3].detach().cpu().numpy(), prob_scalings.detach().cpu().numpy(), colors.detach().cpu().numpy(), res=12)
# o3d.visualization.draw_geometries([scene])
#%%

pose_to_render = np.eye(4)
pose_to_render[:3] = poses[i].detach().cpu().numpy()

opengl_to_opencv = np.array([ 

    [1., 0., 0., 0.],

    [0., -1., 0., 0.],

    [0., 0., -1., 0.],

    [0., 0., 0., 1.]

])

pose_to_render = pose_to_render @ opengl_to_opencv
pose_to_render = np.linalg.inv(pose_to_render)

H, W, K = splat.get_camera_intrinsics()

def custom_draw_geometry_with_custom_fov(pcd):

    vis = o3d.visualization.Visualizer()

    vis.create_window(width=W.item(), height=H.item())

    vis.add_geometry(pcd)

    ctr = vis.get_view_control()

    intr = o3d.camera.PinholeCameraIntrinsic(W.item(), H.item(), K.cpu().numpy())
    params = o3d.camera.PinholeCameraParameters()
    params.intrinsic = intr
    params.extrinsic = pose_to_render
    asdf = ctr.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)

    if asdf:
        vis.run()

        image = vis.capture_screen_float_buffer()
        image = np.asarray(image)
        cv2.imwrite(f'ellipsoids_{i}.png', cv2.cvtColor( image * 255, cv2.COLOR_BGR2RGB))
        vis.destroy_window()
    else:
        print("Failed to set camera parameters")

custom_draw_geometry_with_custom_fov(scene)


#%%
# o3d.io.write_triangle_mesh('scene.obj', scene, compressed=True, write_vertex_colors=True, write_vertex_normals=True, write_triangle_uvs=True, print_progress=True)

# %%
