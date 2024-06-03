#%%
import torch
from splat.utils import GSplat
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path_to_gsplat = 'flightroom_gaussians_sparse_deep.json'

tnow = time.time()
gsplat = GSplat(path_to_gsplat, device, kdtree=True)
print('Time to load GSplat:', time.time() - tnow)
#%%
tnow = time.time()
h, grad_h = gsplat.query_distance(torch.rand(3).to(device), radius=0.1)
print('Time to query distance:', time.time() - tnow)
# %%
