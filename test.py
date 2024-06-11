
#%%
import open3d as o3d
import torch
from splat.utils import GSplat
from cbf.utils import CBF
from dynamics.systems import DoubleIntegrator, SingleIntegrator
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path_to_gsplat = 'flightroom_gaussians_sparse_deep.json'
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
x = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=device).to(torch.float32)
xf = torch.tensor([0.35, 0.09, 0.0, 0.0, 0.0, 0.0], device=device).to(torch.float32)

u_des = 0.1*(xf[:3] - x[:3])
# u_des = torch.tensor([0.1, 0.0, 0.0], device=device).to(torch.float32)
dt = 0.05
traj = [x]

for i in range(500):
    tnow = time.time()

    # catch value error if the QP is infeasible 
    try:
        u = cbf.solve_QP(x, u_des)
    except ValueError:
        print('QP is infeasible')
        break
    # u = cbf.solve_QP(x, u_des)
    print('Control input:', u)
    # print state
    print(f"State: {x}")
    # x = 0.1*u + x
    # lets append three 0s to the end of u
    u = torch.cat((torch.zeros(3).to(u.device), u))
    x = dynamics.system(x, u)[0] + u*dt + x
    traj.append(x)
    print('Time to solve CBF QP:', time.time() - tnow)

#%%
traj = torch.stack(traj)

#%% Save trajectory
import json

data = {
    'traj': traj.cpu().numpy().tolist()
}

with open('traj.json', 'w') as f:
    json.dump(data, f, indent=4)

# print(traj)
# %%

# print(u_des)

# # lets plot the trajectory using matplotlib
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# traj = traj.cpu().numpy()

# ax.plot(traj[:, 0], traj[:, 1], traj[:, 2])
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')

# plt.show()