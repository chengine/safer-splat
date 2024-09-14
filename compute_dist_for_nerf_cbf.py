#%%
import torch
from pathlib import Path    
from splat.gsplat_utils import GSplatLoader
import time
import numpy as np
from tqdm import tqdm
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

method = 'nerf_cbf'
for scene_name in ['statues', 'flight', 'stonehenge', 'old_union']:

    if scene_name == 'old_union':
        radius = 0.01
        path_to_gsplat = Path('outputs/old_union2/splatfacto/2024-09-02_151414/config.yml')

    elif scene_name == 'stonehenge':
        radius = 0.015
        path_to_gsplat = Path('outputs/stonehenge/splatfacto/2024-09-11_100724/config.yml')

    elif scene_name == 'statues':  
        radius = 0.03
        path_to_gsplat = Path('outputs/statues/splatfacto/2024-09-11_095852/config.yml')

    elif scene_name == 'flight':
        radius = 0.03
        path_to_gsplat = Path('outputs/flight/splatfacto/2024-09-12_172434/config.yml')

    nerf_cbf_data_path = f'trajs/{scene_name}_{method}.json'

    print(f"Running {scene_name} with {method}")

    tnow = time.time()
    gsplat = GSplatLoader(path_to_gsplat, device)
    print('Time to load GSplat:', time.time() - tnow)

    # Load data
    with open(nerf_cbf_data_path) as f:
        dataset = json.load(f)

    total_data = dataset['total_data']

    for j in range(len(total_data)):
        data = total_data[j]

        traj = torch.tensor(data['traj']).to(device)
        safety = []
        for i in tqdm(range(len(traj)), desc=f"Simulating trajectory {j}"):
            pt = traj[i, :3]
            # find the distance to the gsplat
            h, grad_h, hess_h, info = gsplat.query_distance(pt, radius=radius, distance_type=None)
            # record min value of h
            safety.append(torch.min(h).item())

        data['safety'] = safety

        total_data[j] = data

    # Save trajectory
    dataset['total_data'] = total_data
    with open(f'trajs/{scene_name}_{method}.json', 'w') as f:
        json.dump(dataset, f, indent=4)

# %%
