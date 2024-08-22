#%%
import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy
from ellipsoids.gs_utils import quaternion_to_rotation_matrix
import time

device = torch.device('cuda')


# def real_get_root(r0, r1, z0, z1, z2, g, max_iterations=100):
#     n0 = r0 * z0
#     n1 = r1 * z1
#     s0 = z2 - 1.

#     if g < 0:
#         s1 = 0.
#     else:
#         s1 = np.sqrt(n0**2 + n1**2 + z2**2) - 1.
    
#     s = 0
#     for i in range(max_iterations):
#         s = (s0 + s1) / 2

#         ratio0 = n0 / (s + r0)
#         ratio1 = n1 / (s + r1)
#         ratio2 = z2 / (s + 1.)
        
#         g = ratio0**2 + ratio1**2 + ratio2**2 - 1
        
#         if g >= 0:
#             s0 = s
#         elif g < 0:
#             s1 = s

#     return s

# def distance_point_ellipsoid(e0, e1, e2, y0, y1, y2):

#     # e0, e1, e2: semi-axes of the ellipsoid (e0 > e1 > e2)

#     z0 = y0 / e0
#     z1 = y1 / e1
#     z2 = y2 / e2
#     g = z0**2 + z1**2 + z2**2 - 1.

#     r0 = (e0 / e2) ** 2
#     r1 = (e1 / e2) ** 2
#     sb = real_get_root(r0, r1, z0, z1, z2, g)
    
#     x0 = r0 * y0 / (sb + r0)
#     x1 = r1 * y1 / (sb + r1)
#     x2 = y2 / (sb + 1)
    
#     distance = np.sqrt((x0 - y0)**2 + (x1 - y1)**2 + (x2-y2)**2)
     
#     return distance, (x0, x1, x2)

def real_get_root(r, z, g, max_iterations=100):
    n = r*z

    s = torch.zeros((len(n), 2), device=device)
    s[:, 0] = z[..., -1] - 1.
    g_pos = (g >= 0)
    s[g_pos, 1] = torch.linalg.norm(n[g_pos], dim=-1) - 1.
    
    for i in range(max_iterations):
        s_i = torch.mean(s, dim=-1, keepdims=True)
        ratio = n / (s_i + r)
        g = torch.sum(ratio**2, dim=-1) - 1
        
        g_pos = (g >= 0)

        # s[:, 0][g_pos] = s_i[g_pos]
        # s[:, 1][~g_pos] = s_i[~g_pos]

        s_i = s_i.squeeze()
        s[g_pos, 0] = s_i[g_pos]
        s[~g_pos, 1] = s_i[~g_pos]
        # if g >= 0:
        #     s0 = s
        # elif g < 0:
        #     s1 = s

    return s_i

def distance_point_ellipsoid(e, y):

    # e0, e1, e2: semi-axes of the ellipsoid (e0 > e1 > e2)
    z = y / e
    g = torch.sum(z**2, dim=-1) - 1.
    r = (e[..., :] / e[..., -2:-1])**2

    sb = real_get_root(r, z, g)
    
    x = r * y / (sb[..., None] + r)
    
    distance = torch.linalg.norm(x - y, dim=-1)
     
    return distance, x

# torch.manual_seed(0)
n = 1000000
dim = 3

quats = torch.rand(n, 4).cuda()
quats = quats / torch.linalg.norm(quats, dim=-1, keepdim=True)
rot_A = quaternion_to_rotation_matrix(quats)
scale_A = 5e-2 + 0.01*torch.rand(n, 3).cuda()
sqrt_sigma_A = torch.bmm(rot_A, torch.diag_embed(scale_A))
Sigma_A = torch.bmm(sqrt_sigma_A, sqrt_sigma_A.transpose(2, 1))
scale_A = scale_A**2

Sigma_A = Sigma_A[:, :dim, :dim]
mu_A = torch.rand(n, dim).cuda()

mu_B = torch.rand(len(scale_A), dim, device=device)

ind, es = torch.sort(torch.sqrt(scale_A), dim=-1)[::-1]

tnow = time.time()
dist, closest_x = distance_point_ellipsoid(es, mu_B)
print('Bisection: ', time.time() - tnow)
print(dist)
# %%

# import cvxpy as cvx

# tnow = time.time()
# dists = []
# for scale_A_, mu_B_ in zip(scale_A[:10], mu_B[:10]):
#     x = cvx.Variable(dim)

#     objective = cvx.Minimize(cvx.norm(x - mu_B_.cpu().numpy(),2))
#     constraints = [cvx.quad_form(x, torch.inverse(torch.diag(scale_A_)).cpu().numpy()) <= 1]

#     prob = cvx.Problem(objective, constraints)
#     prob.solve()

#     # print(x.value)
#     # print(prob.value)
#     dists.append(prob.value)
# print('CVX program: ', time.time() - tnow)
# print(dists)

#%%