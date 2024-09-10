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
import open3d as o3d
import torch
from pathlib import Path    
from splat.gsplat_utils import GSplatLoader
from cbf.cbf_utils import CBF
from dynamics.systems import DoubleIntegrator, SingleIntegrator
import time
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# path_to_gsplat = 'flightroom_gaussians_sparse_deep.json'
path_to_gsplat = Path('outputs/old_union2/splatfacto/2024-09-02_151414/config.yml')

alpha = 5
beta = 5
radius = 0.01
tnow = time.time()
gsplat = GSplatLoader(path_to_gsplat, device)

print('Time to load GSplat:', time.time() - tnow)

dynamics = DoubleIntegrator(device=device, ndim=3)
#%%
tnow = time.time()
cbf = CBF(gsplat, dynamics, alpha, beta, radius)
print('Time to initialize CBF:', time.time() - tnow)
# %%

# For flightroom
# x = torch.tensor([0.0, 0.1, 0.05, 0.0, 0.0, 0.0], device=device).to(torch.float32)
# x = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=device).to(torch.float32)

# x = torch.tensor([0.0, 0.0, 0.0, 0.0, 0n.0, 0.0], device=device).to(torch.float32)
# xf = torch.tensor([0.5, 0.09, -0.04, 0.0, 0.0, 0.0], device=device).to(torch.float32)

# For old union
# x = torch.tensor([-0.1, 0.47, -0.17, 0.0, 0.0, 0.0], device=device).to(torch.float32)
# xf = torch.tensor([0.35, -0.2, -0.18, 0.0, 0.0, 0.0], device=device).to(torch.float32)

x = torch.tensor([0.19, 0.47, -0.17, 0.0, 0.0, 0.0], device=device).to(torch.float32)
xf = torch.tensor([-0.28, -0.2, -0.14, 0.0, 0.0, 0.0], device=device).to(torch.float32)

dt = 0.05
traj = [x]
times = [0]
u_values = []
u_des_values = []
safety = []

for i in tqdm(range(500), desc="Simulating trajectory"):

    vel_des = 5.0*(xf[:3] - x[:3])
    vel_des = torch.clamp(vel_des, -0.1, 0.1)

    # add d term
    vel_des = vel_des + 1.0*(xf[3:] - x[3:])
    # cap between -0.1 and 0.1
    u_des = 1.0*(vel_des - x[3:])
    # cap between -0.1 and 0.1
    u_des = torch.clamp(u_des, -0.1, 0.1)

    tnow = time.time()

    u = cbf.solve_QP(x, u_des)
   
    x_ = x
    x = double_integrator_dynamics(x,u)*dt + x

    # It's gotten stuck
    if torch.norm(x - x_) < 0.0001:
        break

    traj.append(x)
    times.append((i+1) * dt)
    u_values.append(u.cpu().numpy())
    u_des_values.append(u_des.cpu().numpy())

    # let's also record the first manhalanobis distance
    h, grad_h, hess_h, info = gsplat.query_distance(x, radius=radius)
    # record min value of h
    safety.append(torch.min(h).item())

    # print('Time to solve CBF QP:', time.time() - tnow)

#%%
traj = torch.stack(traj)
u_values = np.array(u_values)
u_des_values = np.array(u_des_values)
#%% Save trajectory
import json

data = {
    'traj': traj.cpu().numpy().tolist()
}

with open('traj.json', 'w') as f:
    json.dump(data, f, indent=4)

#%% Plot trajectory vs time
traj_np = traj.cpu().numpy()


fig, axs = plt.subplots(3, 1, figsize=(14, 12))

for ax in axs:
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=12)

axs[0].plot(times, traj_np[:, 0], label='X', color='b', linewidth=2)
axs[0].set_ylabel('X', fontsize=16)
axs[0].legend(fontsize=14)

axs[1].plot(times, traj_np[:, 1], label='Y', color='g', linewidth=2)
axs[1].set_ylabel('Y', fontsize=16)
axs[1].legend(fontsize=14)

axs[2].plot(times, traj_np[:, 2], label='Z', color='r', linewidth=2)
axs[2].set_ylabel('Z', fontsize=16)
axs[2].legend(fontsize=14)

plt.xlabel('Time', fontsize=16)
plt.show()

#%% Plot control inputs u and u_des
fig, axs = plt.subplots(3, 2, figsize=(14, 12))

for ax in axs.flatten():
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=12)

labels = ['Ux', 'Uy', 'Uz']

for i in range(3):
    axs[i, 0].plot(times[1:], u_values[:, i], label=labels[i] + ' (u)', color='b', linewidth=2)
    axs[i, 0].set_ylabel(labels[i] + ' (u)', fontsize=16)
    axs[i, 0].legend(fontsize=14)

    axs[i, 1].plot(times[1:], u_des_values[:, i], label=labels[i] + ' (u_des)', color='g', linewidth=2)
    axs[i, 1].set_ylabel(labels[i] + ' (u_des)', fontsize=16)
    axs[i, 1].legend(fontsize=14)

plt.xlabel('Time', fontsize=16)
plt.show()

#%% Plot safety values
fig, ax = plt.subplots(figsize=(14, 8))

# Plot safety values

ax.plot(times[1:], safety, label='Safety Value', color='m', linewidth=2)
ax.axhline(y=0, color='r', linestyle='--', linewidth=2, label='y = 0')
ax.set_ylabel('Safety Value', fontsize=16)
ax.set_xlabel('Time', fontsize=16)
ax.legend(fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=14)
ax.tick_params(axis='both', which='minor', labelsize=12)
plt.show()

#%% Plot 3D trajectory with spherical obstacle
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

ax.plot(traj_np[:, 0], traj_np[:, 1], traj_np[:, 2], linewidth=2)

# # Plot spherical obstacle
# u, v = np.mgrid[0:2*np.pi:100j, 0:np.pi:50j]
# x_sphere = np.cos(u) * np.sin(v)
# y_sphere = np.sin(u) * np.sin(v)
# z_sphere = np.cos(v)
# ax.plot_surface(x_sphere, y_sphere, z_sphere, color='r', alpha=0.3)

# ax.set_xlabel('X', fontsize=16)
# ax.set_ylabel('Y', fontsize=16)
# ax.set_zlabel('Z', fontsize=16)
# ax.tick_params(axis='both', which='major', labelsize=14)
# ax.tick_params(axis='both', which='minor', labelsize=12)

# plt.show()

#%%