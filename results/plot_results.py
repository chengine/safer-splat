#%%
import os
import numpy as np
import json
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

def adjust_lightness(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

scene_names = ['stonehenge', 'statues', 'flight', 'old_union']
methods = ['ball-to-ellipsoid', 'ball-to-ball-squared', 'ball-to-pt-squared', 'nerf_cbf']      # TODO: Add baseline and point cloud methods

metrics = ['Success Rate', 'Min. Distance', 'Effort Difference', 'Ground-Truth Distance']
fig, ax = plt.subplots(2, 2, figsize=(15, 10), dpi=200)

font = {
        'family': 'Arial',
        'weight' : 'bold',
        'size'   : 20}

matplotlib.rc('font', **font)
for i, scene_name in enumerate(scene_names):
    for j, method in enumerate(methods):
        save_fp = str(Path(os.getcwd()).parent.absolute()) + f'/trajs/{scene_name}_{method}.json'
        with open(save_fp, 'r') as f:
            meta = json.load(f)

        radius = meta['radius']
        datas = meta['total_data']

        if method == 'ball-to-ellipsoid':
            col = '#34A853'
            linewidth= 3
 
            traj = []
            u_out = []
            u_des = []
            time_step = []
            sucess = []
            safety = []
            total_time = []
            non_degenerate = 0
            for data in datas:
                # Only store these if more than 2 points
                if len(data['traj']) > 2:
                    traj.append(data['traj'])
                    u_out.append(data['u_out'])
                    u_des.append(data['u_des'])
                    safety.append(np.array(data['safety']).min())
                    non_degenerate += 1

                time_step.append(data['time_step'])
                sucess.append(data['sucess'])
                total_time.append(data['total_time'])
            print('num non-degenerate:', non_degenerate, 'scene', scene_name)
            try:
                traj = np.concatenate(traj)
                u_out = np.concatenate(u_out)
                u_des = np.concatenate(u_des)
                u_diff = np.linalg.norm(u_out - u_des, axis=1)
                u_diff *= 100
            except:
                u_diff = None

            safety = np.stack(safety)
            safety *= 100
            time_step = np.concatenate(time_step)
            sucess = np.concatenate(sucess)
            total_time = np.concatenate(total_time)

            print(f'{scene_name}_{method}')
            print('Num of successes:', sucess.sum(), 'Total time:', total_time.mean())

            # traj = np.concatenate([np.array(data['traj']) for data in datas])
            # u_out = np.concatenate([np.array(data['u_out']) for data in datas])
            # u_des = np.concatenate([np.array(data['u_des']) for data in datas])
            # time_step = np.concatenate([np.array(data['time_step']) for data in datas])
            # safety = np.stack([np.array(data['safety']).min() for data in datas]).squeeze()
            # sucess = np.concatenate([np.array(data['sucess']) for data in datas])
            total_time = np.concatenate([np.array(data['total_time']) for data in datas])
            cbf_solve_time = np.concatenate([np.array(data['cbf_solve_time']) for data in datas])
            qp_solve_time = np.concatenate([np.array(data['qp_solve_time']) for data in datas])
            prune_time = np.concatenate([np.array(data['prune_time']) for data in datas])

            print(f'{scene_name}_{method}')
            print('Min safety:', safety.min(), 'Num of successes:', sucess.sum(), 'Total time:', total_time.mean())
            print('correction', u_diff.mean())

        elif method == 'ball-to-ball-squared':
            col = '#4285F4'
            linewidth=3

            traj = []
            u_out = []
            u_des = []
            time_step = []
            sucess = []
            safety = []
            total_time = []
            non_degenerate = 0
            for data in datas:
                # Only store these if more than 2 points
                if len(data['traj']) > 2:
                    traj.append(data['traj'])
                    u_out.append(data['u_out'])
                    u_des.append(data['u_des'])
                    safety.append(np.array(data['safety']).min())
                    non_degenerate += 1

                time_step.append(data['time_step'])
                sucess.append(data['sucess'])
                total_time.append(data['total_time'])
            print('num non-degenerate:', non_degenerate, 'scene', scene_name)
            try:
                traj = np.concatenate(traj)
                u_out = np.concatenate(u_out)
                u_des = np.concatenate(u_des)
                u_diff = np.linalg.norm(u_out - u_des, axis=1)
                u_diff *= 100
            except:
                u_diff = None

            safety = np.stack(safety)
            safety *= 100
            time_step = np.concatenate(time_step)
            sucess = np.concatenate(sucess)
            total_time = np.concatenate(total_time)

            print(f'{scene_name}_{method}')
            print('Num of successes:', sucess.sum(), 'Total time:', total_time.mean())

            # traj = np.concatenate([np.array(data['traj']) for data in datas])
            # u_out = np.concatenate([np.array(data['u_out']) for data in datas])
            # u_des = np.concatenate([np.array(data['u_des']) for data in datas])
            # time_step = np.concatenate([np.array(data['time_step']) for data in datas])
            # safety = np.stack([np.array(data['safety']).min() for data in datas]).squeeze()
            # sucess = np.concatenate([np.array(data['sucess']) for data in datas])
            total_time = np.concatenate([np.array(data['total_time']) for data in datas])
            cbf_solve_time = np.concatenate([np.array(data['cbf_solve_time']) for data in datas])
            qp_solve_time = np.concatenate([np.array(data['qp_solve_time']) for data in datas])
            prune_time = np.concatenate([np.array(data['prune_time']) for data in datas])

            print(f'{scene_name}_{method}')
            print('Min safety:', safety.min(), 'Num of successes:', sucess.sum(), 'Total time:', total_time.mean())
            print('correction', u_diff.mean())

        elif method == 'ball-to-pt-squared':
            col = '#FBBc05'
            linewidth=3

            traj = []
            u_out = []
            u_des = []
            time_step = []
            sucess = []
            safety = []
            total_time = []
            non_degenerate = 0
            for data in datas:
                # Only store these if more than 2 points
                if len(data['traj']) > 2:
                    traj.append(data['traj'])
                    u_out.append(data['u_out'])
                    u_des.append(data['u_des'])
                    safety.append(np.array(data['safety']).min())
                    non_degenerate += 1

                time_step.append(data['time_step'])
                sucess.append(data['sucess'])
                total_time.append(data['total_time'])
            print('num non-degenerate:', non_degenerate, 'scene', scene_name)
            try:
                traj = np.concatenate(traj)
                u_out = np.concatenate(u_out)
                u_des = np.concatenate(u_des)
                u_diff = np.linalg.norm(u_out - u_des, axis=1)
                u_diff *= 100
            except:
                u_diff = None

            safety = np.stack(safety)
            safety *= 100
            time_step = np.concatenate(time_step)
            sucess = np.concatenate(sucess)
            total_time = np.concatenate(total_time)

            print(f'{scene_name}_{method}')
            print('Num of successes:', sucess.sum(), 'Total time:', total_time.mean())

            # traj = np.concatenate([np.array(data['traj']) for data in datas])
            # u_out = np.concatenate([np.array(data['u_out']) for data in datas])
            # u_des = np.concatenate([np.array(data['u_des']) for data in datas])
            # time_step = np.concatenate([np.array(data['time_step']) for data in datas])
            # safety = np.stack([np.array(data['safety']).min() for data in datas]).squeeze()
            # sucess = np.concatenate([np.array(data['sucess']) for data in datas])
            total_time = np.concatenate([np.array(data['total_time']) for data in datas])
            cbf_solve_time = np.concatenate([np.array(data['cbf_solve_time']) for data in datas])
            qp_solve_time = np.concatenate([np.array(data['qp_solve_time']) for data in datas])
            prune_time = np.concatenate([np.array(data['prune_time']) for data in datas])

            print(f'{scene_name}_{method}')
            print('Min safety:', safety.min(), 'Num of successes:', sucess.sum(), 'Total time:', total_time.mean())
            print('correction', u_diff.mean())

        elif method == 'nerf_cbf':
            col = '#EA4335'
            linewidth=3

            traj = []
            u_out = []
            u_des = []
            time_step = []
            sucess = []
            safety = []
            total_time = []
            non_degenerate = 0
            for data in datas:

                # Only store these if more than 2 points
                if len(data['traj']) > 2:
                    traj.append(data['traj'])
                    u_out.append(data['u_out'])
                    u_des.append(data['u_des'])
                    safety.append(np.array(data['safety']).min())
                    non_degenerate += 1

                time_step.append(data['time_step'])
                sucess.append(data['sucess'])
                total_time.append(data['total_time'])
            print('num non-degenerate:', non_degenerate, 'scene', scene_name)
            try:
                traj = np.concatenate(traj)
                u_out = np.concatenate(u_out)
                u_des = np.concatenate(u_des)
                u_diff = np.linalg.norm(u_out - u_des, axis=1)
                u_diff *= 100
                u_diff /= 0.05
            except:
                u_diff = None

            try:
                safety = np.stack(safety)
                safety *= 100
            except:
                safety = None
            print(safety)
            time_step = np.concatenate(time_step)
            sucess = np.concatenate(sucess)
            total_time = np.concatenate(total_time)

            print(f'{scene_name}_{method}')
            print('Num of successes:', sucess.sum(), 'Total time:', total_time.mean())

        if method == 'nerf_cbf':
            ax[0, 0].bar(i + 0.75*j/len(methods) + 0.25/2, total_time.mean(), width=0.15, color=col, capsize=10, hatch=marking, edgecolor='black', linewidth=linewidth, 
                     linestyle='-', joinstyle='round', rasterized=True)

        else:
            ax[0, 0].bar(i + 0.75*j/len(methods) + 0.25/2, qp_solve_time.mean(), width=0.15, color= adjust_lightness(col, 0.5), linewidth=3, ec='k', label='qp')
            ax[0, 0].bar(i + 0.75*j/len(methods) + 0.25/2, cbf_solve_time.mean(), bottom=qp_solve_time.mean(), width=0.15, color=adjust_lightness(col, 1.0), linewidth=3, ec='k', label='cbf')
            ax[0, 0].bar(i + 0.75*j/len(methods) + 0.25/2, prune_time.mean(), bottom=cbf_solve_time.mean() + qp_solve_time.mean(), width=0.15, color = adjust_lightness(col, 1.), linewidth=3, hatch='x', ec='k', label='prune')

        if safety is not None:
            marking = None
            errors = np.abs(safety.mean().reshape(-1, 1) - np.array([safety.min(), safety.max()]).reshape(-1, 1))

            ax[0, 1].errorbar(i + 0.75*j/len(methods) + 0.25/2, safety.mean().reshape(-1, 1), yerr=errors, color=col, capsize=15, elinewidth=5, alpha=0.5)
            ax[0, 1].scatter( np.repeat((i + 0.75*j/len(methods) + 0.25/2), len(safety)), safety, s=250, color=col, alpha=0.02)
            ax[0, 1].scatter(i +  + 0.75*j/len(methods) + 0.25/2 - 0.13, safety.mean(), s=200, color=col, alpha=1, marker='>')
            
        if u_diff is not None:
            marking = None
            ax[1, 0].bar(i + 0.75*j/len(methods) + 0.25/2, u_diff.mean(), width=0.15, color=col, capsize=10, hatch=marking, edgecolor='black', linewidth=linewidth, 
                        linestyle='-', joinstyle='round', rasterized=True)

            ax[1, 1].bar(i + 0.75*j/len(methods) + 0.25/2, sucess.sum()/100, width=0.15, color=col, capsize=10, hatch=marking, edgecolor='black', linewidth=linewidth, 
                        linestyle='-', joinstyle='round', rasterized=True)

ax[1, 1].set_title(r'Success Rate $\uparrow$', fontsize=25, fontweight='bold')
ax[1, 1].get_xaxis().set_visible(False)
ax[1, 1].grid(axis='y', linewidth=2, color='k', linestyle='--', alpha=0.5, zorder=0)
ax[1, 1].set_axisbelow(True)
for location in ['left', 'right', 'top', 'bottom']:
    ax[1, 1].spines[location].set_linewidth(4)
# ax[0, 0].legend(['ball-to-ellipsoid', 'ball-to-ball-squared'])

ax[0, 1].set_title('Minimum Distance', fontsize=25, fontweight='bold')
ax[0, 1].get_xaxis().set_visible(False)
ax[0, 1].axhline(y = 0., color = 'k', linestyle = '--', linewidth=3, alpha=0.7, zorder=0) 
ax[0, 1].grid(axis='y', linewidth=2, color='k', linestyle='--', alpha=0.5, zorder=0)
ax[0, 1].set_axisbelow(True)
for location in ['left', 'right', 'top', 'bottom']:
    ax[0, 1].spines[location].set_linewidth(4)
ax[0, 1].set_ylim(-.15, 1.)

ax[1, 0].set_title(r'Control Difference $\downarrow$', fontsize=25, fontweight='bold')
ax[1, 0].get_xaxis().set_visible(False)
ax[1, 0].grid(axis='y', linewidth=2, color='k', linestyle='--', alpha=0.5, zorder=0)
ax[1, 0].set_axisbelow(True)
for location in ['left', 'right', 'top', 'bottom']:
    ax[1, 0].spines[location].set_linewidth(4)
ax[1, 0].set_ylim(0, 10.)
# ax[1, 0].set_yscale('log')

ax[0, 0].set_title(r'Computation Time (s) $\downarrow$' , fontsize=25, fontweight='bold')
ax[0, 0].get_xaxis().set_visible(False)
ax[0, 0].grid(axis='y', linewidth=2, color='k', linestyle='--', alpha=0.5, zorder=0)
ax[0, 0].set_axisbelow(True)
for location in ['left', 'right', 'top', 'bottom']:
    ax[0, 0].spines[location].set_linewidth(4)
ax[0,0].set_yscale('log')

# x_axis = np.array([116446, 201584, 281756, 525734, 1690536]) / 1e6
plt.savefig(f'simulation_stats.png', dpi=500)

#%%