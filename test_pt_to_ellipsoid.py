#%%
import torch
from ellipsoids.math_utils import *
from gs_utils.gs_utils import quaternion_to_rotation_matrix
import matplotlib.pyplot as plt
import scipy

# torch.manual_seed(0)
n = 2
dim = 2

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

mu_B = torch.rand(1, dim).cuda()

