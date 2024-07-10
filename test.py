
# #%%
# import open3d as o3d
# import torch
# from splat.utils import GSplat
# from cbf.utils import CBF
# from dynamics.systems import DoubleIntegrator, SingleIntegrator
# import time

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# # path_to_gsplat = 'flightroom_gaussians_sparse_deep.json'
# path_to_gsplat = 'gaussians_test.json'
# alpha = lambda x: 1.0 * x
# tnow = time.time()
# gsplat = GSplat(path_to_gsplat, device, kdtree=True)

# print('Time to load GSplat:', time.time() - tnow)

# dynamics = DoubleIntegrator(device=device, ndim=3)
# #%%
# tnow = time.time()
# cbf = CBF(gsplat, dynamics, alpha)
# print('Time to initialize CBF:', time.time() - tnow)
# # %%
# x = torch.tensor([-2.5, 0.05, 0.0, 0.1, 0.0, 0.0], device=device).to(torch.float32)
# xf = torch.tensor([0.0, 0.25, 0.0, 0.0, 0.0, 0.0], device=device).to(torch.float32)


# # u_des = torch.tensor([0.0, 0.0, 0.0], device=device).to(torch.float32)
# dt = 0.05
# traj = [x]

# for i in range(200):

#     vel_des = 0.1*(xf[:3] - x[:3])
#     vel_des = torch.clamp(vel_des, -0.1, 0.1)

#     # add d term
#     vel_des = vel_des + 5.0*(xf[3:] - x[3:])
#     # cap between -0.1 and 0.1
#     u_des = 0.1*(vel_des - x[3:])
#     # cap between -0.1 and 0.1
#     u_des = torch.clamp(u_des, -0.1, 0.1)

#     tnow = time.time()

#     # catch value error if the QP is infeasible 
#     try:
#         u = cbf.solve_QP(x, u_des)
#     except ValueError:
#         print('QP is infeasible')
#         break
#     # u = cbf.solve_QP(x, u_des)
#     # u = u_des
#     # clip u between .2 and -.2
#     u = torch.clamp(u, -0.2, 0.2)
#     print('Control input:', u)
#     # print state
#     print(f"State: {x[:3]}")
#     # x = 0.1*u + x
#     # lets append three 0s to the end of u
#     u = torch.cat((torch.zeros(3).to(u.device), u))
#     x = dynamics.system(x, u)[0] + u*dt + x
#     traj.append(x)
#     # print('Time to solve CBF QP:', time.time() - tnow)

# #%%
# traj = torch.stack(traj)

# #%% Save trajectory
# import json

# data = {
#     'traj': traj.cpu().numpy().tolist()
# }

# with open('traj.json', 'w') as f:
#     json.dump(data, f, indent=4)

# # print(traj)
# # %%

# # print(u_des)

# # # lets plot the trajectory using matplotlib
# # import matplotlib.pyplot as plt
# # from mpl_toolkits.mplot3d import Axes3D

# # fig = plt.figure()
# # ax = fig.add_subplot(111, projection='3d')

# # traj = traj.cpu().numpy()

# # ax.plot(traj[:, 0], traj[:, 1], traj[:, 2])
# # ax.set_xlabel('X')
# # ax.set_ylabel('Y')
# # ax.set_zlabel('Z')

# # plt.show()



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
from splat.utils import GSplat
from cbf.utils import CBF
from dynamics.systems import DoubleIntegrator, SingleIntegrator
import time
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path_to_gsplat = 'flightroom_gaussians_sparse_deep.json'
# path_to_gsplat = 'gaussians_test.json'
alpha = lambda x: 1.0 * x
tnow = time.time()
gsplat = GSplat(path_to_gsplat, device, kdtree=True)

print('Time to load GSplat:', time.time() - tnow)

dynamics = DoubleIntegrator(device=device, ndim=3)
#%%
tnow = time.time()
cbf = CBF(gsplat, dynamics, alpha)
print('Time to initialize CBF:', time.time() - tnow)
# %%
x = torch.tensor([0.0, 0.1, 0.05, 0.0, 0.0, 0.0], device=device).to(torch.float32)
# x0 = 0

# x = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=device).to(torch.float32)
xf = torch.tensor([0.5, 0.1, 0.05, 0.0, 0.0, 0.0], device=device).to(torch.float32)

dt = 0.05
traj = [x]
times = [0]
u_values = []
u_des_values = []
safety = []

for i in tqdm(range(250), desc="Simulating trajectory"):

    vel_des = 5.0*(xf[:3] - x[:3])
    vel_des = torch.clamp(vel_des, -0.1, 0.1)

    # add d term
    vel_des = vel_des + 1.0*(xf[3:] - x[3:])
    # cap between -0.1 and 0.1
    u_des = 1.0*(vel_des - x[3:])
    # cap between -0.1 and 0.1
    u_des = torch.clamp(u_des, -0.1, 0.1)

    # u_des = torch.tensor([0.1, 0.0, 0.0], device=device).to(torch.float32)


    tnow = time.time()

    u = cbf.solve_QP(x, u_des)
   
    x = double_integrator_dynamics(x,u)*dt + x

    traj.append(x)
    times.append((i+1) * dt)
    u_values.append(u.cpu().numpy())
    u_des_values.append(u_des.cpu().numpy())

    # let's also record the first manhalanobis distance
    h, grad_h, hes_h = gsplat.query_distance(x)
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

#%% Plot safety values

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