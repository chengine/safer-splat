#%%
import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy
from ellipsoids.covariance_utils import quaternion_to_rotation_matrix
import time
from splat.gsplat_utils import DummyGSplatLoader
import cvxpy as cvx
from scipy.spatial.transform import Rotation
from ellipsoids.plot_utils import plot_ellipse

device = torch.device('cuda')

#%%
torch.manual_seed(2)
n = 3
dim = 2

if dim == 3:
    quats = torch.rand(n, 4).cuda()
    quats = quats / torch.linalg.norm(quats, dim=-1, keepdim=True)
    rot_A = quaternion_to_rotation_matrix(quats)

    scale_A = torch.rand(n, dim).cuda()

else:
    rot = Rotation.from_euler('Z', np.pi * np.random.rand(n))
    rot_A = torch.tensor(rot.as_matrix(), device=device, dtype=torch.float32)
    quats = torch.tensor(rot.as_quat(), device=device, dtype=torch.float32)
    quats = torch.roll(quats, 1, dims=-1)

    scale_A = torch.rand(n, 3).cuda()
    scale_A[:, -1] = 0.

sqrt_sigma_A = torch.bmm(rot_A, torch.diag_embed(scale_A))
Sigma_A = torch.bmm(sqrt_sigma_A, sqrt_sigma_A.transpose(2, 1))

mu_A = torch.rand(n, 3) - 1.
mu_B = torch.rand(3, device=device)

if dim == 2:
    mu_A[:, -1] = 0.
    mu_B[-1] = 0.

splat = DummyGSplatLoader(device)
splat.initialize_attributes(mu_A, quats, scale_A)

#%%

radius = 0.05
tnow = time.time()
h, grad_h, hess_h, info = splat.query_distance(mu_B, radius=radius)
print('Bisection: ', time.time() - tnow)

#%%
y_preds = info['y'].cpu().numpy()
phis = info['phi'].cpu().numpy()

for i, (Sigma_A_, mu_A_) in enumerate( zip( Sigma_A, mu_A ) ):
    y = cvx.Variable(dim)

    objective = cvx.Minimize( cvx.sum_squares(y - mu_B[:dim].cpu().numpy()) )
    constraints = [cvx.quad_form( (y - mu_A_[:dim].cpu().numpy()), np.linalg.inv(Sigma_A_[:dim, :dim].cpu().numpy())) <= 1.]

    prob = cvx.Problem(objective, constraints)
    prob.solve()

    yopt = y.value
    dist = phis[i]*prob.value - (radius)**2

    print('-------------------------------------')
    print( 'Difference in distances', np.abs( dist - h[i].item() ) )
    print( 'Difference in closest point', np.linalg.norm( yopt - y_preds[i][:dim] ) ) 
    print('-------------------------------------')



#%% Plotting

fig, ax = plt.subplots(1, 1, figsize=(10, 10))

kwargs = {'facecolor': 'red', 'edgecolor': None, 'alpha': 0.5}
for i, (Sigma_A_, mu_A_) in enumerate( zip( Sigma_A, mu_A ) ):
    plot_ellipse(mu_A_[:2], Sigma_A_[:2, :2], 1., ax, **kwargs)

ax.set_xlim([-2., 2.])
ax.set_ylim([-2., 2.])
plt.show()
# %%
