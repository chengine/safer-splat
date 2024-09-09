#%%
import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy
from ellipsoids.covariance_utils import quaternion_to_rotation_matrix
import time
from splat.gsplat_utils import DummyGSplatLoader
import cvxpy as cvx

device = torch.device('cuda')

#%%
# torch.manual_seed(0)
n = 1000000
dim = 3

quats = torch.rand(n, 4).cuda()
quats = quats / torch.linalg.norm(quats, dim=-1, keepdim=True)
rot_A = quaternion_to_rotation_matrix(quats)
scale_A = torch.rand(n, 3).cuda()
sqrt_sigma_A = torch.bmm(rot_A, torch.diag_embed(scale_A))
Sigma_A = torch.bmm(sqrt_sigma_A, sqrt_sigma_A.transpose(2, 1))

mu_A = torch.rand(n, dim)
mu_B = torch.rand(dim, device=device)

scales, inds = torch.sort(scale_A, dim=-1, descending=True)

splat = DummyGSplatLoader(device)
splat.initialize_attributes(mu_A, quats, scales)

#%%

radius = 0.05
tnow = time.time()
h, grad_h, hess_h, info = splat.query_distance(mu_B, radius=radius)
print('Bisection: ', time.time() - tnow)

y_preds = info['y'].cpu().numpy()
phis = info['phi'].cpu().numpy()

num_to_compare = 10
tnow = time.time()
for i, (Sigma_A_, mu_A_) in enumerate( zip( Sigma_A[:num_to_compare], mu_A[:num_to_compare] ) ):
    y = cvx.Variable(dim)

    objective = cvx.Minimize( cvx.sum_squares(y - mu_B.cpu().numpy()) )
    constraints = [cvx.quad_form( (y - mu_A_.cpu().numpy()), np.linalg.inv(Sigma_A_.cpu().numpy())) <= 1.]

    prob = cvx.Problem(objective, constraints)
    prob.solve()

    yopt = y.value
    dist = phis[i]*prob.value - (radius)**2

    print( 'Difference in distances', np.abs( dist - h[i].item() ) )
    print( 'Difference in closest point', np.linalg.norm( yopt - y_preds[i] ) )

print('CVX program: ', time.time() - tnow)

#%%