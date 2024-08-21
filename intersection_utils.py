#%%
import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy
from ellipsoids.gs_utils import quaternion_to_rotation_matrix

def real_get_root(r0, r1, z0, z1, z2, g, max_iterations=100):
    n0 = r0 * z0
    n1 = r1 * z1
    s0 = z2 - 1.

    if g < 0:
        s1 = 0.
    else:
        s1 = n0**2 + n1**2 + z2**2 - 1.
    
    s = 0
    for _ in range(max_iterations):
        s = (s0 + s1) / 2
        if s == s0 or s == s1:
            break
        
        ratio0 = n0 / (s + r0)
        ratio1 = n1 / (s + r1)
        ratio2 = z2 / (s + 1.)
        
        g = ratio0**2 + ratio1**2 + ratio2**2 - 1
        
        if g > 0:
            s0 = s
        elif g < 0:
            s1 = s
        else:
            break
    return s

def distance_point_ellipsoid(e0, e1, e2, y0, y1, y2):
    z0 = y0 / e0
    z1 = y1 / e1
    z2 = y2 / e2
    g = z0**2 + z1**2 + z2**2 - 1.

    r0 = (e0 / e2) ** 2
    r1 = (e1 / e2) ** 2
    sb = real_get_root(r0, r1, z0, z1, z2, g)
    
    x0 = r0 * y0 / (sb + r0)
    x1 = r1 * y1 / (sb + r1)
    x2 = y2 / (sb + 1)
    
    distance = np.sqrt((x0 - y0)**2 + (x1 - y1)**2 + (x2-y2)**2)
     
    return distance, (x0, x1, x2)

# torch.manual_seed(0)
n = 2
dim = 3

device = torch.device('cuda')

quats = torch.rand(n, 4).cuda()
quats = quats / torch.linalg.norm(quats, dim=-1, keepdim=True)
rot_A = quaternion_to_rotation_matrix(quats)
scale_A = 5e-2 + 0.01*torch.rand(n, 3).cuda()
sqrt_sigma_A = torch.bmm(rot_A, torch.diag_embed(scale_A))
Sigma_A = torch.bmm(sqrt_sigma_A, sqrt_sigma_A.transpose(2, 1))
scale_A = scale_A**2

Sigma_A = Sigma_A[:, :dim, :dim]
mu_A = torch.rand(n, dim).cuda()

mu_B = torch.rand(1, dim).cpu().numpy()

es = np.sqrt(scale_A[0].cpu().numpy())

dist, closest_x = distance_point_ellipsoid(es[0], es[1], es[2], mu_B[0, 0], mu_B[0, 1], mu_B[0, 2])
# %%

import cvxpy as cvx

x = cvx.Variable(dim)

objective = cvx.Minimize(cvx.norm(x - mu_B[0],2))
constraints = [cvx.quad_form(x - mu_A[0].cpu().numpy(), torch.inverse(Sigma_A[0]).cpu().numpy()) <= 1]

prob = cvx.Problem(objective, constraints)
prob.solve()

print(x.value)

#%%