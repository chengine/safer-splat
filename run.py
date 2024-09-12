#%%
def double_integrator_dynamics(x, u):
    """
    Returns the dynamics (xdot) for a 3-dimensional double integrator system.
    Parameters:
    x (torch.Tensor): State vector [x, y, z, vx, vy, vz]
    u (torch.Tensor): Input vector [ux, uy, uz]

    Returns:
    torch.Tensor: The derivative of the state vector [vx, vy, vz, ax, ay, az]
    """
    assert x.shape == (6,), "State vector x must be of shape (6,)"
    assert u.shape == (3,), "Input vector u must be of shape (3,)"

    # The state vector x consists of position (x, y, z) and velocity (vx, vy, vz)
    pos = x[:3]
    vel = x[3:]

    # The input vector u consists of accelerations (ax, ay, az)
    acc = u

    # The derivative of the state vector is the velocity and acceleration
    xdot = torch.cat((vel, acc))

    return xdot

#%%
import torch
from pathlib import Path    
from splat.gsplat_utils import GSplatLoader
from cbf.cbf_utils import CBF
from dynamics.systems import DoubleIntegrator
import time
import numpy as np
from tqdm import tqdm
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# path_to_gsplat = Path('outputs/old_union2/splatfacto/2024-09-02_151414/config.yml')
# path_to_gsplat = Path('outputs/statues/splatfacto/2024-09-11_095852/config.yml')
# path_to_gsplat = Path('outputs/stonehenge/splatfacto/2024-09-11_100724/config.yml')

alpha = 5.
beta = 1.
radius = 0.015
dt = 0.05

n = 100
n_steps = 500

t = np.linspace(0, 2*np.pi, n)
t_z = 10*np.linspace(0, 2*np.pi, n)

# scene_name = 'old_union'
# method = 'ball-to-ellipsoid'
# method = 'ball-to-ball-squared'
# method = 'mahalanobis'

for scene_name in ['stonehenge']:
    for method in ['ball-to-ellipsoid', 'ball-to-ball-squared', 'mahalanobis']:

        if scene_name == 'old_union':
            radius_z = 0.01
            radius_config = 1.35/2
            mean_config = np.array([0.14, 0.23, -0.15])
            path_to_gsplat = Path('outputs/old_union2/splatfacto/2024-09-02_151414/config.yml')

        elif scene_name == 'stonehenge':
            radius_z = 0.01
            radius_config = 0.784/2
            mean_config = np.array([-0.08, -0.03, 0.05])
            path_to_gsplat = Path('outputs/stonehenge/splatfacto/2024-09-11_100724/config.yml')

        elif scene_name == 'statues':
            radius_z = 0.03    
            radius_config = 0.475
            mean_config = np.array([-0.064, -0.0064, -0.025])
            path_to_gsplat = Path('outputs/statues/splatfacto/2024-09-11_095852/config.yml')

        print(f"Running {scene_name} with {method}")

        tnow = time.time()
        gsplat = GSplatLoader(path_to_gsplat, device)
        print('Time to load GSplat:', time.time() - tnow)

        dynamics = DoubleIntegrator(device=device, ndim=3)
    
        ### Create configurations

        # For flightroom
        # x = torch.tensor([0.0, 0.1, 0.05, 0.0, 0.0, 0.0], device=device).to(torch.float32)
        # x = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=device).to(torch.float32)

        # x = torch.tensor([0.0, 0.0, 0.0, 0.0, 0n.0, 0.0], device=device).to(torch.float32)
        # xf = torch.tensor([0.5, 0.09, -0.04, 0.0, 0.0, 0.0], device=device).to(torch.float32)

        # For old union
        # x = torch.tensor([-0.1, 0.47, -0.17, 0.0, 0.0, 0.0], device=device).to(torch.float32)
        # xf = torch.tensor([0.35, -0.2, -0.14, 0.0, 0.0, 0.0], device=device).to(torch.float32)

        # x = torch.tensor([0.19, 0.47, -0.17, 0.0, 0.0, 0.0], device=device).to(torch.float32)
        # xf = torch.tensor([-0.28, -0.2, -0.14, 0.0, 0.0, 0.0], device=device).to(torch.float32)

        x0 = np.stack([radius_config*np.cos(t), radius_config*np.sin(t), radius_z * np.sin(t_z)], axis=-1)     # starting positions
        x0 = x0 + mean_config

        xf = np.stack([radius_config*np.cos(t + np.pi), radius_config*np.sin(t + np.pi), radius_z * np.sin(t_z + np.pi)], axis=-1)     # goal positions
        xf = xf + mean_config

        # Run simulation
        total_data = []

        for trial, (start, goal) in enumerate(zip(x0, xf)):
            x = torch.tensor(start).to(device).to(torch.float32)
            x = torch.cat([x, torch.zeros(3).to(device).to(torch.float32)])
            goal = torch.tensor(goal).to(device).to(torch.float32)
            goal = torch.cat([goal, torch.zeros(3).to(device).to(torch.float32)])
            traj = [x]
            times = [0]
            u_values = []
            u_des_values = []
            safety = []
            sucess = []

            total_time = []

            cbf = CBF(gsplat, dynamics, alpha, beta, radius, distance_type=method)
            for i in tqdm(range(n_steps), desc=f"Simulating trajectory {trial}"):

                vel_des = 5.0*(goal[:3] - x[:3])
                vel_des = torch.clamp(vel_des, -0.1, 0.1)

                # add d term
                vel_des = vel_des + 1.0*(goal[3:] - x[3:])
                # cap between -0.1 and 0.1
                u_des = 1.0*(vel_des - x[3:])
                # cap between -0.1 and 0.1
                u_des = torch.clamp(u_des, -0.1, 0.1)

                tnow = time.time()
                torch.cuda.synchronize()
                u = cbf.solve_QP(x, u_des)
                torch.cuda.synchronize()
                total_time.append(time.time() - tnow)

                x_ = x
                x = double_integrator_dynamics(x,u)*dt + x

                traj.append(x)
                times.append((i+1) * dt)
                u_values.append(u.cpu().numpy())
                u_des_values.append(u_des.cpu().numpy())

                # record some stuff
                h, grad_h, hess_h, info = gsplat.query_distance(x, radius=radius)
                # record min value of h
                safety.append(torch.min(h).item())

                # It's gotten stuck
                if torch.norm(x - x_) < 0.0001:
                    if torch.norm(x - goal) < 0.001:
                        print("Reached Goal")
                        sucess.append(True)
                    else:
                        sucess.append(False)
                    break

            if i >= n_steps - 1:
                sucess.append(True)

            traj = torch.stack(traj)
            u_values = np.array(u_values)
            u_des_values = np.array(u_des_values)

            data = {
            'traj': traj.cpu().numpy().tolist(),
            'u_out': u_values.tolist(),
            'u_des': u_des_values.tolist(),
            'time_step': times,
            'safety': safety,
            'sucess': sucess,
            'total_time': total_time,
            'cbf_solve_time': cbf.times_cbf,
            'qp_solve_time': cbf.times_qp,
            'prune_time': cbf.times_prune,
            }

            total_data.append(data)

        # Save trajectory
        data = {
            'scene': scene_name,
            'method': method,
            'alpha': alpha,
            'beta': beta,
            'radius': radius,
            'dt': dt,
            'total_data': total_data,
        }

        with open(f'trajs/{scene_name}_{method}.json', 'w') as f:
            json.dump(data, f, indent=4)

# %%
