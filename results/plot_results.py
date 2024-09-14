#%%
import os
import numpy as np
import json
import open3d as o3d
import matplotlib
import matplotlib.pyplot as plt
import time
from pathlib import Path
from matplotlib.patches import FancyBboxPatch
import pandas as pd

# scene_name = 'old_union'
# method = 'ball-to-ellipsoid'
# method = 'ball-to-ball-squared'
# method = 'mahalanobis'

scene_names = ['stonehenge', 'statues', 'flight', 'old_union']
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
# for i, scene_name in enumerate(scene_names):
#     for j, method in enumerate(methods):
#         save_fp = str(Path(os.getcwd()).parent.absolute()) + f'/trajs/{scene_name}_{method}.json'
#         with open(save_fp, 'r') as f:
#             meta = json.load(f)

#         radius = meta['radius']
#         datas = meta['total_data']

#         traj = np.concatenate([np.array(data['traj']) for data in datas])
#         u_out = np.concatenate([np.array(data['u_out']) for data in datas])
#         u_des = np.concatenate([np.array(data['u_des']) for data in datas])
#         time_step = np.concatenate([np.array(data['time_step']) for data in datas])
#         safety = np.stack([np.array(data['safety']).min() for data in datas]).squeeze()
#         sucess = np.concatenate([np.array(data['sucess']) for data in datas])
#         total_time = np.concatenate([np.array(data['total_time']) for data in datas])
#         cbf_solve_time = np.concatenate([np.array(data['cbf_solve_time']) for data in datas])
#         qp_solve_time = np.concatenate([np.array(data['qp_solve_time']) for data in datas])
#         prune_time = np.concatenate([np.array(data['prune_time']) for data in datas])

#         u_diff = np.linalg.norm(u_out - u_des, axis=1)
#         print(f'{scene_name}_{method}')
#         print('Min safety:', safety.min(), 'Num of successes:', sucess.sum(), 'Total time:', total_time.mean())
#         print('correction', u_diff.mean())

#         if method == 'ball-to-ellipsoid':
#             # marking = 'o'
#             col = '#34A853'
#             linewidth=3

#         elif method == 'ball-to-ball-squared':
#             #marking = 'x'
#             col = '#4285F4'

#             linewidth=3

#         elif method == 'point-cloud':
#             col = '#FBBc05'
#             linewidth=3

#         elif method == 'nerf':
#             col = '#EA4335'
#             linewidth=3

#         if scene_name == 'stonehenge':
#             #col = '#34A853'
#             # marking = 'o'
#             marking = None

#         elif scene_name == 'statues':
#             #col = '#4285F4'
#             # marking = 'x'
#             marking = None

#         elif scene_name == 'old_union':
#             #col = '#FBBc05'
#             # marking = 's'
#             marking = None

#         elif scene_name == 'flight':
#             #col = '#EA4335'
#             # marking = '*'
#             marking = None

#         safety *= 100
#         u_diff *= 100

#         errors = np.abs(safety.mean().reshape(-1, 1) - np.array([safety.min(), safety.max()]).reshape(-1, 1))

#         ax[0].bar(i + 0.75*j/len(methods) + 0.25/2, sucess.sum()/100, width=0.3, color=col, capsize=10, hatch=marking, edgecolor='black', linewidth=linewidth, 
#                      linestyle='-', joinstyle='round', rasterized=True)
        
#         ax[2].errorbar(i + 0.75*j/len(methods) + 0.25/2, safety.mean().reshape(-1, 1), yerr=errors, color=col, capsize=10, elinewidth=5, alpha=0.5)
#         ax[2].scatter( np.repeat((i + 0.75*j/len(methods) + 0.25/2), len(safety)), safety, s=250, color=col, alpha=0.03)
#         ax[2].scatter(i +  + 0.75*j/len(methods) + 0.25/2 - 0.13, safety.mean(), s=200, color=col, alpha=1, marker='>')

#         ax[1].bar(i + 0.75*j/len(methods) + 0.25/2, u_diff.mean(), width=0.3, color=col, capsize=10, hatch=marking, edgecolor='black', linewidth=linewidth, 
#                      linestyle='-', joinstyle='round', rasterized=True)
        
# ax[0].set_title(r'Success Rate $\uparrow$', fontsize=15, fontweight='bold')
# ax[0].get_xaxis().set_visible(False)
# for location in ['left', 'right', 'top', 'bottom']:
#     ax[0].spines[location].set_linewidth(4)
# # ax[0, 0].legend(['ball-to-ellipsoid', 'ball-to-ball-squared'])

# ax[2].set_title('Min. Distance', fontsize=15, fontweight='bold')
# ax[2].get_xaxis().set_visible(False)
# ax[2].axhline(y = 0., color = 'r', linestyle = '--', alpha=0.3) 
# for location in ['left', 'right', 'top', 'bottom']:
#     ax[2].spines[location].set_linewidth(4)

# ax[1].set_title(r'Control Difference $\downarrow$', fontsize=15, fontweight='bold')
# ax[1].get_xaxis().set_visible(False)
# for location in ['left', 'right', 'top', 'bottom']:
#     ax[1].spines[location].set_linewidth(4)

# plt.savefig(f'simulation_stats.pdf')

#%%

font = {
        'family': 'Arial',
        'weight' : 'bold',
        'size'   : 20}
matplotlib.rc('font', **font)

def adjust_lightness(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

scene_names = ['stonehenge', 'flight', 'statues', 'old_union']

fig, ax = plt.subplots(1, figsize=(15, 10), dpi=200)
x_axis = np.array([116446, 201584, 281756, 525734, 1690536]) / 1e6

bar_width = 0.025

for j, method in enumerate(methods):

    total_times = []
    cbf_solve_times = []
    qp_solve_times = []
    prune_times = []

    for i, scene_name in enumerate(scene_names):
        save_fp = str(Path(os.getcwd()).parent.absolute()) + f'/trajs/{scene_name}_{method}.json'
        with open(save_fp, 'r') as f:
            meta = json.load(f)

        datas = meta['total_data']

        total_time = np.concatenate([np.array(data['total_time']) for data in datas])
        cbf_solve_time = np.concatenate([np.array(data['cbf_solve_time']) for data in datas])
        qp_solve_time = np.concatenate([np.array(data['qp_solve_time']) for data in datas])
        prune_time = np.concatenate([np.array(data['prune_time']) for data in datas])

        if method == 'ball-to-ellipsoid':
            # marking = 'o'
            col = '#34A853'
            linewidth=3
            offset = -0.5*bar_width

        elif method == 'ball-to-ball-squared':
            #marking = 'x'
            col = '#4285F4'
            linewidth=3
            offset = 0.5*bar_width

        # elif method == 'point-cloud':
        #     col = '#FBBc05'
        #     linewidth=3

        # elif method == 'nerf':
        #     col = '#EA4335'
        #     linewidth=3

        total_times.append(total_time.mean())
        cbf_solve_times.append(cbf_solve_time.mean())
        qp_solve_times.append(qp_solve_time.mean())
        prune_times.append(prune_time.mean())

    total_times = np.array(total_times)
    cbf_solve_times = np.array(cbf_solve_times)
    qp_solve_times = np.array(qp_solve_times)
    prune_times = np.array(prune_times)

    ax.plot(x_axis[:-1], (qp_solve_times+ cbf_solve_times + prune_times), color=col, linewidth=6, marker='o', label=method)

    ax.bar(x_axis[:-1]+offset, qp_solve_times, width=bar_width, alpha=0.5, hatch='o', color= adjust_lightness(col, 0.5), linewidth=3, label='qp')
    ax.bar(x_axis[:-1]+offset, cbf_solve_times, bottom=qp_solve_times, width=bar_width, alpha=0.5, color=adjust_lightness(col, 0.75), hatch='o', linewidth=3, ec='k', label='cbf')
    ax.bar(x_axis[:-1]+offset, prune_times, bottom=cbf_solve_times + qp_solve_times, width=bar_width, alpha=0.5, color = adjust_lightness(col, 1.), hatch='x', linewidth=3, ec='k', label='prune')
    # ax.bar(x_axis[:-1]+offset, total_times - prune_times, bottom=prune_times, width=bar_width, alpha=0.5, color=adjust_lightness(col, 0.8), linewidth=3, ec='k', label='total')

# ax[0].set_title('Success Rate (Higher is better)', fontsize=20, fontweight='bold')
for location in ['left', 'right', 'top', 'bottom']:
    ax.spines[location].set_linewidth(4)
ax.set_ylabel('Time (s)', fontsize=25, fontweight='bold')
# ax.legend()
plt.savefig(f'timings.pdf')

#%%