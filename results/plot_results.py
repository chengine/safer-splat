#%%
import os
import numpy as np
import json
import open3d as o3d
import matplotlib
import matplotlib.pyplot as plt
import time
from pathlib import Path

# scene_name = 'old_union'
# method = 'ball-to-ellipsoid'
# method = 'ball-to-ball-squared'
# method = 'mahalanobis'

scene_names = ['stonehenge', 'statues', 'old_union', 'flight']
methods = ['ball-to-ellipsoid', 'ball-to-ball-squared']      # TODO: Add baseline and point cloud methods

# data = {
#     'scene': scene_name,
#     'method': method,
#     'alpha': alpha,
#     'beta': beta,
#     'radius': radius,
#     'dt': dt,
#     'total_data': total_data,
# }

# mesh_fp = f'{name}.ply'

# mesh_ = o3d.io.read_triangle_mesh(mesh_fp)
# # mesh_ = mesh_.filter_smooth_taubin(number_of_iterations=100)
# mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh_)

metrics = ['Success Rate', 'Min. Distance', 'Effort Difference', 'Ground-Truth Distance']
fig, ax = plt.subplots(1,3, figsize=(15, 5), dpi=200)

font = {
        'family': 'Arial',
        'weight' : 'bold',
        'size'   : 15}

matplotlib.rc('font', **font)

for i, scene_name in enumerate(scene_names):
    for j, method in enumerate(methods):
        save_fp = str(Path(os.getcwd()).parent.absolute()) + f'/trajs/{scene_name}_{method}.json'
        with open(save_fp, 'r') as f:
            meta = json.load(f)

        radius = meta['radius']
        datas = meta['total_data']

        traj = np.concatenate([np.array(data['traj']) for data in datas])
        u_out = np.concatenate([np.array(data['u_out']) for data in datas])
        u_des = np.concatenate([np.array(data['u_des']) for data in datas])
        time_step = np.concatenate([np.array(data['time_step']) for data in datas])
        safety = np.stack([np.array(data['safety']).min() for data in datas]).squeeze()
        sucess = np.concatenate([np.array(data['sucess']) for data in datas])
        total_time = np.concatenate([np.array(data['total_time']) for data in datas])
        cbf_solve_time = np.concatenate([np.array(data['cbf_solve_time']) for data in datas])
        qp_solve_time = np.concatenate([np.array(data['qp_solve_time']) for data in datas])
        prune_time = np.concatenate([np.array(data['prune_time']) for data in datas])

        u_diff = np.linalg.norm(u_out - u_des, axis=1)
        print(f'{scene_name}_{method}')
        print('Min safety:', safety.min(), 'Num of successes:', sucess.sum(), 'Total time:', total_time.mean())
        print('correction', u_diff.mean())

        if method == 'ball-to-ellipsoid':
            # marking = 'o'
            col = '#34A853'
            linewidth=3

        elif method == 'ball-to-ball-squared':
            #marking = 'x'
            col = '#EA4335'

            linewidth=3

        if scene_name == 'stonehenge':
            #col = '#34A853'
            marking = 'o'

        elif scene_name == 'statues':
            #col = '#4285F4'
            marking = 'x'
            
        elif scene_name == 'old_union':
            #col = '#FBBc05'
            marking = 's'

        elif scene_name == 'flight':
            #col = '#EA4335'
            marking = '*'

        safety *= 100
        u_diff *= 100

        errors = np.abs(safety.mean().reshape(-1, 1) - np.array([safety.min(), safety.max()]).reshape(-1, 1))

        ax[0].bar(i + 0.75*j/len(methods) + 0.25/2, sucess.sum()/100, width=0.3, color=col, capsize=10, hatch=marking, edgecolor='black', linewidth=linewidth, 
                     linestyle='-', joinstyle='round', rasterized=True)
        
        ax[2].errorbar(i + 0.75*j/len(methods) + 0.25/2, safety.mean().reshape(-1, 1), yerr=errors, color=col, capsize=10, elinewidth=5, alpha=0.5)
        ax[2].scatter( np.repeat((i + 0.75*j/len(methods) + 0.25/2), len(safety)), safety, s=250, color=col, alpha=0.03)
        ax[2].scatter(i +  + 0.75*j/len(methods) + 0.25/2 - 0.13, safety.mean(), s=200, color=col, alpha=1, marker='>')

        ax[1].bar(i + 0.75*j/len(methods) + 0.25/2, u_diff.mean(), width=0.3, color=col, capsize=10, hatch=marking, edgecolor='black', linewidth=linewidth, 
                     linestyle='-', joinstyle='round', rasterized=True)
        
ax[0].set_title('Success Rate (Higher is better)', fontsize=15, fontweight='bold')
ax[0].get_xaxis().set_visible(False)
for location in ['left', 'right', 'top', 'bottom']:
    ax[0].spines[location].set_linewidth(4)
# ax[0, 0].legend(['ball-to-ellipsoid', 'ball-to-ball-squared'])

ax[2].set_title('Min. Distance', fontsize=15, fontweight='bold')
ax[2].get_xaxis().set_visible(False)
ax[2].axhline(y = 0., color = 'r', linestyle = '--', alpha=0.3) 
for location in ['left', 'right', 'top', 'bottom']:
    ax[2].spines[location].set_linewidth(4)

ax[1].set_title('Control Difference (Lower is better)', fontsize=15, fontweight='bold')
ax[1].get_xaxis().set_visible(False)
for location in ['left', 'right', 'top', 'bottom']:
    ax[1].spines[location].set_linewidth(4)

plt.savefig(f'simulation_stats.pdf')

#%%

fig, ax = plt.subplots(1, figsize=(15, 10), dpi=200)

for i, scene_name in enumerate(scene_names):
    for j, method in enumerate(methods):
        save_fp = str(Path(os.getcwd()).parent.absolute()) + f'/trajs/{scene_name}_{method}.json'
        with open(save_fp, 'r') as f:
            meta = json.load(f)

        datas = meta['total_data']

        total_time = np.concatenate([np.array(data['total_time']) for data in datas])
        cbf_solve_time = np.concatenate([np.array(data['cbf_solve_time']) for data in datas])
        qp_solve_time = np.concatenate([np.array(data['qp_solve_time']) for data in datas])
        prune_time = np.concatenate([np.array(data['prune_time']) for data in datas])

        if method == 'ball-to-ellipsoid':
            col = "green"
            linewidth=3

        elif method == 'ball-to-ball-squared':
            col = "blue"
            linewidth=1

        if scene_name == 'stonehenge':
            marking = '/'

        elif scene_name == 'statues':
            marking = 'x'

        elif scene_name == 'old_union':
            marking = 'o'

        elif scene_name == 'flight':
            marking = '*'

        ax[0].plot()
        
# ax[0].set_title('Success Rate (Higher is better)', fontsize=20, fontweight='bold')
for location in ['left', 'right', 'top', 'bottom']:
    ax[0].spines[location].set_linewidth(4)

plt.savefig(f'timings.pdf')