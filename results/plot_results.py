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

scene_names = ['stonehenge', 'statues', 'flight', 'old_union', 'flight-low-res']
methods = ['ball-to-ellipsoid', 'ball-to-pt-squared', 'ball-to-ball-squared', 'nerf_cbf']      # TODO: Add baseline and point cloud methods

n = 100

t = np.linspace(0, 2*np.pi, n)
t_z = 10*np.linspace(0, 2*np.pi, n)

metrics = ['Success Rate', 'Min. Distance', 'Effort Difference', 'Ground-Truth Distance']
fig, ax = plt.subplots(2, 2, figsize=(15, 10), dpi=200)

font = {
        'family': 'Arial',
        'weight' : 'bold',
        'size'   : 20}

matplotlib.rc('font', **font)
for k, scene_name in enumerate(scene_names):

    if scene_name == 'old_union':
        radius_z = 0.01
        radius = 0.01
        radius_config = 1.35/2
        mean_config = np.array([0.14, 0.23, -0.15])

    elif scene_name == 'stonehenge':
        radius_z = 0.01
        radius = 0.015
        radius_config = 0.784/2
        mean_config = np.array([-0.08, -0.03, 0.05])

    elif scene_name == 'statues':
        radius_z = 0.03    
        radius = 0.03
        radius_config = 0.475
        mean_config = np.array([-0.064, -0.0064, -0.025])

    elif scene_name == 'flight':
        radius_z = 0.06
        radius = 0.03
        radius_config = 0.545/2
        mean_config = np.array([0.19, 0.01, -0.02])

    x0 = np.stack([radius_config*np.cos(t), radius_config*np.sin(t), radius_z * np.sin(t_z)], axis=-1)     # starting positions
    x0 = x0 + mean_config

    xf = np.stack([radius_config*np.cos(t + np.pi), radius_config*np.sin(t + np.pi), radius_z * np.sin(t_z + np.pi)], axis=-1)     # goal positions
    xf = xf + mean_config

    for j, method in enumerate(methods):
        save_fp = str(Path(os.getcwd()).parent.absolute()) + f'/trajs/{scene_name}_{method}.json'

        if scene_name == 'flight-low-res' and method == 'nerf_cbf':
            continue

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
            feasible = []
            dist_to_goal = []
            safety = []
            total_time = []
            non_degenerate = 0
            for i, data in enumerate(datas):
                goal = xf[i]
                start = x0[i]
                # Only store these if more than 2 points
                if len(data['traj']) > 2:

                    traj.append(data['traj'])
                    dist_to_goal.append(1. - np.linalg.norm(np.array(data['traj'])[:, :3] - goal, axis=-1).min() / np.linalg.norm(goal - start))

                    u_out.append(data['u_out'])
                    u_des.append(data['u_des'])
                    safety.append(np.array(data['safety']).min())
                    non_degenerate += 1

                time_step.append(data['time_step'])
                feasible.append(data['feasible'])
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

            dist_to_goal = np.stack(dist_to_goal)

            safety = np.stack(safety)
            safety *= 100

            time_step = np.concatenate(time_step)
            feasible = np.concatenate(feasible)
            total_time = np.concatenate(total_time)

            print(f'{scene_name}_{method}')
            print('Num of successes:', feasible.sum(), 'Total time:', total_time.mean())

            # traj = np.concatenate([np.array(data['traj']) for data in datas])
            # u_out = np.concatenate([np.array(data['u_out']) for data in datas])
            # u_des = np.concatenate([np.array(data['u_des']) for data in datas])
            # time_step = np.concatenate([np.array(data['time_step']) for data in datas])
            # safety = np.stack([np.array(data['safety']).min() for data in datas]).squeeze()
            # feasible = np.concatenate([np.array(data['feasible']) for data in datas])
            total_time = np.concatenate([np.array(data['total_time']) for data in datas])
            cbf_solve_time = np.concatenate([np.array(data['cbf_solve_time']) for data in datas])
            qp_solve_time = np.concatenate([np.array(data['qp_solve_time']) for data in datas])
            prune_time = np.concatenate([np.array(data['prune_time']) for data in datas])
            print('prune time', prune_time.mean(), 'qp time', qp_solve_time.mean(), 'cbf time', cbf_solve_time.mean())

            #print(f'{scene_name}_{method}')
            #print('Min safety:', safety.min(), 'Num of successes:', feasible.sum(), 'Total time:', total_time.mean())
            #print('correction', u_diff.mean())

        elif method == 'ball-to-ball-squared':
            col = '#4285F4'
            linewidth=3

            traj = []
            u_out = []
            u_des = []
            time_step = []
            feasible = []
            dist_to_goal = []
            safety = []
            total_time = []
            non_degenerate = 0
            for i, data in enumerate(datas):
                goal = xf[i]
                start = x0[i]
                # Only store these if more than 2 points
                if len(data['traj']) > 2:

                    traj.append(data['traj'])
                    dist_to_goal.append(1. - np.linalg.norm(np.array(data['traj'])[:, :3] - goal, axis=-1).min() / np.linalg.norm(goal - start))

                    u_out.append(data['u_out'])
                    u_des.append(data['u_des'])
                    safety.append(np.array(data['safety']).min())
                    non_degenerate += 1

                time_step.append(data['time_step'])
                feasible.append(data['feasible'])
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

            dist_to_goal = np.stack(dist_to_goal)

            safety = np.stack(safety)
            safety *= 100

            time_step = np.concatenate(time_step)
            feasible = np.concatenate(feasible)
            total_time = np.concatenate(total_time)

            print(f'{scene_name}_{method}')
            print('Num of successes:', feasible.sum(), 'Total time:', total_time.mean())

            # traj = np.concatenate([np.array(data['traj']) for data in datas])
            # u_out = np.concatenate([np.array(data['u_out']) for data in datas])
            # u_des = np.concatenate([np.array(data['u_des']) for data in datas])
            # time_step = np.concatenate([np.array(data['time_step']) for data in datas])
            # safety = np.stack([np.array(data['safety']).min() for data in datas]).squeeze()
            # feasible = np.concatenate([np.array(data['feasible']) for data in datas])
            total_time = np.concatenate([np.array(data['total_time']) for data in datas])
            cbf_solve_time = np.concatenate([np.array(data['cbf_solve_time']) for data in datas])
            qp_solve_time = np.concatenate([np.array(data['qp_solve_time']) for data in datas])
            prune_time = np.concatenate([np.array(data['prune_time']) for data in datas])
            print('prune time', prune_time.mean(), 'qp time', qp_solve_time.mean(), 'cbf time', cbf_solve_time.mean())

            # print(f'{scene_name}_{method}')
            # print('Min safety:', safety.min(), 'Num of successes:', feasible.sum(), 'Total time:', total_time.mean())
            # print('correction', u_diff.mean())

        elif method == 'ball-to-pt-squared':
            col = '#FBBc05'
            linewidth=3

            traj = []
            u_out = []
            u_des = []
            time_step = []
            feasible = []
            dist_to_goal = []
            safety = []
            total_time = []
            non_degenerate = 0
            for i, data in enumerate(datas):
                goal = xf[i]
                start = x0[i]
                # Only store these if more than 2 points
                if len(data['traj']) > 2:
                    traj.append(data['traj'])
                    dist_to_goal.append(1. - np.linalg.norm(np.array(data['traj'])[:, :3] - goal, axis=-1).min() / np.linalg.norm(goal - start))

                    u_out.append(data['u_out'])
                    u_des.append(data['u_des'])
                    safety.append(np.array(data['safety']).min())
                    non_degenerate += 1

                time_step.append(data['time_step'])
                feasible.append(data['feasible'])
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

            dist_to_goal = np.stack(dist_to_goal)

            safety = np.stack(safety)
            safety *= 100

            time_step = np.concatenate(time_step)
            feasible = np.concatenate(feasible)
            total_time = np.concatenate(total_time)

            print(f'{scene_name}_{method}')
            print('Num of successes:', feasible.sum(), 'Total time:', total_time.mean())

            # traj = np.concatenate([np.array(data['traj']) for data in datas])
            # u_out = np.concatenate([np.array(data['u_out']) for data in datas])
            # u_des = np.concatenate([np.array(data['u_des']) for data in datas])
            # time_step = np.concatenate([np.array(data['time_step']) for data in datas])
            # safety = np.stack([np.array(data['safety']).min() for data in datas]).squeeze()
            # feasible = np.concatenate([np.array(data['feasible']) for data in datas])
            total_time = np.concatenate([np.array(data['total_time']) for data in datas])
            cbf_solve_time = np.concatenate([np.array(data['cbf_solve_time']) for data in datas])
            qp_solve_time = np.concatenate([np.array(data['qp_solve_time']) for data in datas])
            prune_time = np.concatenate([np.array(data['prune_time']) for data in datas])
            print('prune time', prune_time.mean(), 'qp time', qp_solve_time.mean(), 'cbf time', cbf_solve_time.mean())
            # print(f'{scene_name}_{method}')
            # print('Min safety:', safety.min(), 'Num of successes:', feasible.sum(), 'Total time:', total_time.mean())
            # print('correction', u_diff.mean())

        elif method == 'nerf_cbf':
  
            col = '#EA4335'
            linewidth=3

            traj = []
            u_out = []
            u_des = []
            time_step = []
            feasible = []
            dist_to_goal = []
            safety = []
            total_time = []
            non_degenerate = 0
            for i, data in enumerate(datas):
                goal = xf[i]
                start = x0[i]
                dist_to_goal.append(1. - np.linalg.norm(np.array(data['traj'])[:, :3] - goal, axis=-1).min() / np.linalg.norm(goal - start))

                # Only store these if more than 2 points
                if len(data['traj']) > 2:

                    traj.append(data['traj'])

                    u_out.append(data['u_out'])
                    u_des.append(data['u_des'])
                    safety.append(np.array(data['safety']).min())
                    non_degenerate += 1

                time_step.append(data['time_step'])
                feasible.append(data['sucess'])
                total_time.append(data['total_time'])
            print('num non-degenerate:', non_degenerate, 'scene', scene_name)

            dist_to_goal = np.stack(dist_to_goal)

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

            time_step = np.concatenate(time_step)
            feasible = np.concatenate(feasible)
            total_time = np.concatenate(total_time)

            print(f'{scene_name}_{method}')
            print('Num of successes:', feasible.sum(), 'Total time:', total_time.mean())

        if method == 'nerf_cbf':
            ax[0, 0].bar(k + 0.75*j/len(methods) + 0.25/2, total_time.mean(), width=0.15, color=col, capsize=10, hatch=marking, edgecolor='black', linewidth=linewidth, 
                     linestyle='-', joinstyle='round', rasterized=True)

        else:
            # ax[0, 0].bar(k + 0.75*j/len(methods) + 0.25/2, qp_solve_time.mean(), bottom = 0, width=0.15, color= adjust_lightness(col, 0.5), linewidth=3, ec='k', label='qp')
            # ax[0, 0].bar(k + 0.75*j/len(methods) + 0.25/2, cbf_solve_time.mean(), bottom=qp_solve_time.mean(), width=0.15, color=adjust_lightness(col, 1.0), linewidth=3, ec='k', label='cbf')
            # ax[0, 0].bar(k + 0.75*j/len(methods) + 0.25/2, prune_time.mean(), bottom=cbf_solve_time.mean() + qp_solve_time.mean(), width=0.15, color = adjust_lightness(col, 1.), linewidth=3, hatch='x', ec='k', label='prune')

            ax[0, 0].bar(k + 0.75*j/len(methods) + 0.25/2, prune_time.mean() + cbf_solve_time.mean() + qp_solve_time.mean(), width=0.15, color = adjust_lightness(col, 1.), linewidth=3,  ec='k', label='prune')

        if safety is not None:
            marking = None
            errors = np.abs(safety.mean().reshape(-1, 1) - np.array([safety.min(), safety.max()]).reshape(-1, 1))

            ax[0, 1].errorbar(k + 0.75*j/len(methods) + 0.25/2, safety.mean().reshape(-1, 1), yerr=errors, color=adjust_lightness(col, 0.5), markeredgewidth=5, capsize=15, elinewidth=5, alpha=0.5)
            ax[0, 1].scatter( np.repeat((k + 0.75*j/len(methods) + 0.25/2), len(safety)), safety, s=250, color=col, alpha=0.04)
            ax[0, 1].scatter(k +  + 0.75*j/len(methods) + 0.25/2 - 0.13, safety.mean(), s=200, color=col, alpha=1, marker='>')
            
        if u_diff is not None:
            marking = None
            ax[1, 0].bar(k + 0.75*j/len(methods) + 0.25/2, u_diff.mean(), width=0.15, color=col, capsize=10, hatch=marking, edgecolor='black', linewidth=linewidth, 
                        linestyle='-', joinstyle='round', rasterized=True)

        # ax[1, 1].bar(k + 0.75*j/len(methods) + 0.25/2, feasible.sum()/100, width=0.15, color=col, capsize=10, hatch=marking, edgecolor='black', linewidth=linewidth, 
        #             linestyle='-', joinstyle='round', rasterized=True)

        errors_dist = np.abs(dist_to_goal.mean().reshape(-1, 1) - np.array([dist_to_goal.min(), dist_to_goal.max()]).reshape(-1, 1))

        # ax[1, 1].errorbar(k + 0.75*j/len(methods) + 0.25/2, dist_to_goal.mean().reshape(-1, 1), yerr=errors_dist, color=col, capsize=15, elinewidth=5, alpha=1.)
        # ax[1, 1].scatter( np.repeat((k + 0.75*j/len(methods) + 0.25/2), len(dist_to_goal)), dist_to_goal, s=150, color=col, alpha=0.03)
        # ax[1, 1].scatter(k +  + 0.75*j/len(methods) + 0.25/2 - 0.1, dist_to_goal.mean(), s=200, color=col, alpha=1, marker='>')

        ax[1, 1].bar(k + 0.75*j/len(methods) + 0.25/2, dist_to_goal.mean(), width=0.15, color=col, capsize=10, hatch=marking, edgecolor='black', linewidth=linewidth, 
                    linestyle='-', joinstyle='round', rasterized=True)

# Plot timings for non-minimal and point cloud
ax[0, 0].scatter(np.arange(len(scene_names)-1) + 0.75*(len(methods) - 1)/len(methods) / 2 + 0.25/2, np.array([1/1.5, 1/1.32, 1/1.14, 2.42]), s=200, color='black', marker='x', label='non-minimal', alpha=1)
ax[0, 0].scatter(np.arange(len(scene_names)-1) + 0.75*(len(methods) - 1)/len(methods) / 2 + 0.25/2, np.array([1/5.5, 1/3.3, 1/2.19, 1/1.28]), s=200, color='black', marker='*', label='pointcloud', alpha=1)
ax[0, 0].scatter(np.arange(len(scene_names)-1) + 0.75*(len(methods) - 1)/len(methods) / 2 + 0.25/2, np.array([4.3, 5.3, 6.1, 16.1]), s=200, color='black', marker='+', label='non-minimal pointcloud', alpha=1)

ax[1, 1].set_title(r'Progress to Goal $\uparrow$', fontsize=25, fontweight='bold')
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
ax[0, 1].set_ylim(-.15, .2)

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