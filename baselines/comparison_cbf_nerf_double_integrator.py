# %%
import argparse
import os
import time

import numpy as np
import torch
import cv2
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib
import sys
from pathlib import Path

# append to PATH
sys.path.append(f"{os.path.expanduser('~/Research/utilities/nerf_cbf_controller')}")
sys.path.append(f"{Path(__file__).parent.parent}")

from src import config
from src.tools.viz import SLAMFrontend
from src.utils.datasets import get_dataset
from src.utils.Renderer import Renderer
from src.NICE_SLAM import NICE_SLAM
from src.common import get_camera_from_tensor
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

from utilities.nerf_utils import NeRF
from nerfstudio.cameras.cameras import Cameras, CameraType
import json

# with open('device.txt', encoding='utf-8') as file:
#      device=file.read()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   
   
alpha = 5.
beta = 1.
radius = 0.015
dt = 0.05

n = 100
n_steps = 500

t = np.linspace(0, 2*np.pi, n)
t_z = 10*np.linspace(0, 2*np.pi, n)

# default parameters specified by the paper (for those not overwritten)
time_step = dt
mu = 0
# alpha = 0.5
# safety distance
d = 0.1
u = torch.tensor([[0,0,1]]).t().float().to(device)
max_h = 2
intend = sigma = 1
# beta = 1

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


def state_to_pose(state):
    if state.shape[0] == 3:
        batch_pose = torch.eye(4, device=state.device)
        batch_pose[:3, -1] = state[:3]
        
        return batch_pose
    
    for i in range(state.shape[0]):
        rm = torch.from_numpy(R.from_euler('xyz', state[i][3:].cpu(), degrees=True).as_matrix()).to(device)
        pose = torch.cat((torch.cat((rm, state[i][:3].unsqueeze(-1)), 1), torch.tensor([[0,0,0,1]]).to(device)), 0).unsqueeze(0)
        if i == 0:
            batch_pose = pose
        else:
            batch_pose = torch.cat((batch_pose, pose), dim=0)
    return batch_pose

def update_dynamics(state, v, action):
    return state + time_step * v + 0.5 * action * (time_step ** 2), v + time_step * action


class Robot:
    def __init__(self, config_path, device, cfg=None):
        # initialize NeRF
        self.nerf = NeRF(config_path=config_path,
                         res_factor=None,
                         test_mode="test", #"inference", "val"
                         dataset_mode="val",
                         device=device)
        
        # device
        self.device = device
        
        # initialize a camera
        self.camera = Cameras(
            fx=torch.tensor(60),
            fy=torch.tensor(60),
            cx=torch.tensor(59.95),
            cy=torch.tensor(33.95),
            camera_type=CameraType.PERSPECTIVE,
            width=torch.tensor(68),
            height=torch.tensor(120),
            camera_to_worlds=torch.eye(4, device=self.device)[None, :3, :]
        )
        
        # initialize velocity
        self.v = torch.zeros(6).to(device)

    def predict_observation(self, pose):
        # update the pose of the camera
        self.camera.camera_to_worlds = pose[None, :3, :] if len(pose.shape) == 2 else pose[:, :3, :]
        
        # render the observations
        outputs = self.nerf.render(cameras=self.camera)
        
        # extract the depth
        depth = outputs['depth']
        
        return depth.squeeze()
    
    def render(self, pose):
        # update the intrinsic properties of the cameras
        self.camera.fx = torch.tensor(600)
        self.camera.fy = torch.tensor(600)
        self.camera.cx = torch.tensor(599.5)
        self.camera.cy = torch.tensor(339.5)
        
        # update the width and height of the camera
        self.camera.width = 1200
        self.camera.height = 680
        
        # update the pose of the camera
        self.camera.camera_to_worlds = pose[None, :3, :] if len(pose.shape) == 2 else pose[:, :3, :]
        
        # render the observations
        outputs = self.nerf.render(cameras=self.camera)
        
        # update the intrinsic properties of the cameras
        self.camera.fx = torch.tensor(60)
        self.camera.fy = torch.tensor(60)
        self.camera.cx = torch.tensor(59.95)
        self.camera.cy = torch.tensor(33.95)
        
        # update the width and height of the camera
        self.camera.width = 120
        self.camera.height = 68
        
        return outputs['depth'].squeeze(), outputs['rgb']

def find_safe_action(robot, pos_vel_state, h, intended_action, max_runs=50):
    # state = torch.cat((pose[:3, -1].to(device), torch.from_numpy(R.from_matrix(pose[:3, :3].cpu()).as_euler('xyz', degrees=True)).to(device)), dim=0)
    # alternative:
    # predict the observation
    new_state = double_integrator_dynamics(pos_vel_state.squeeze(), intended_action)*dt + pos_vel_state
    new_state, new_v = new_state[:3], new_state[3:]
    new_state = new_state #.unsqueeze(0)
    new_pose = state_to_pose(new_state)
    new_h = d - robot.predict_observation(new_pose).min().unsqueeze(0) + beta * torch.norm(new_v, p=2)
    
    unsafe_metric = new_h - alpha * h
    
    if unsafe_metric <= 0:
        # print('Intended action {} is safe'.format(orient_action))
        # print('intervention = 0')
        return intended_action, new_h, True
    
    # initial run
    run = 0
    
    print(f"Applying the Safety Filter...")
    
    while run < max_runs:
        new_action = torch.tensor(np.random.normal(intended_action.cpu().numpy(), sigma),
                                  device=device).float()
        # predict the observation
        new_state = double_integrator_dynamics(pos_vel_state.squeeze(), new_action)*dt + pos_vel_state
        new_state, new_v = new_state[:3], new_state[3:]
        new_state = new_state #.unsqueeze(0)
        new_pose = state_to_pose(new_state)
        new_h = d - robot.predict_observation(new_pose).min().unsqueeze(0) + beta * torch.norm(new_v, p=2)
        # print(f'shapes d: {d}, v: {batch_new_v.shape} min. obs: {robot.predict_observation(batch_new_pose).min(dim=-1)[0].min(dim=-1)[0]}')
        batch_new_h = d - robot.predict_observation(new_pose).min(dim=-1)[0].min(dim=-1)[0] + beta * torch.norm(new_v, dim=-1, p=2)
        
        unsafe_metric = batch_new_h - alpha * h
        
        if unsafe_metric <= 0:
            # print('Intended action {} is safe'.format(orient_action))
            # print('intervention = 0')
            return new_action, batch_new_h, True
        
        # increment counter
        run += 1
        
    return intended_action, h, False


if __name__ == '__main__':
    
    # option to visualize the method
    visualize_method: bool = False
    
    # the robot's position state
    robot_pos_state = torch.zeros(6, device=device)
    
    # the robot's velocity state
    robot_vel_state = torch.zeros(6, device=device)
    
    # output path for the data
    data_output_path = Path(f'trajs/cbf_nerf')
    
    # create parent directory, if necessary
    data_output_path.mkdir(exist_ok=True, parents=True)
    
    for scene_name in ['old_union', 'stonehenge', 'statues', 'flight']:
        for method in ['cbf_nerf']:
            try:
                # base path to outputs
                outputs_base_path = Path("~/Research/Gaussian_Splatting/outputs/nerfstudio")
                
                if scene_name == 'old_union':
                    radius_z = 0.01
                    radius_config = 1.35/2
                    mean_config = np.array([0.14, 0.23, -0.15])
                    # config path
                    config_path = Path(f"{os.path.expanduser(f'{outputs_base_path}/old_union2/nerfacto/2024-09-12_203602/config.yml')}")
                elif scene_name == 'stonehenge':
                    radius_z = 0.01
                    radius_config = 0.784/2
                    mean_config = np.array([-0.08, -0.03, 0.05])
                    # config path
                    config_path = Path(f"{os.path.expanduser(f'{outputs_base_path}/stonehenge/nerfacto/2024-09-12_211002/config.yml')}")
                elif scene_name == 'statues':
                    radius_z = 0.03    
                    radius_config = 0.475
                    mean_config = np.array([-0.064, -0.0064, -0.025])
                    # config path
                    # config_path = Path(f"{os.path.expanduser(f'{outputs_base_path}/statues/nerfacto/2024-09-12_204832/config.yml')}")
                    config_path = Path(f"{os.path.expanduser(f'{outputs_base_path}/statues/nerfacto/2024-09-14_095328/config.yml')}")
                elif scene_name == 'flight':
                    radius_z = 0.06
                    radius_config = 0.545/2
                    mean_config = np.array([0.19, 0.01, -0.02])
                    # config path
                    config_path = Path(f"{os.path.expanduser(f'{outputs_base_path}/flight/nerfacto/2024-09-14_083406/config.yml')}")

                print(f"Running {scene_name} with {method}")

                tnow = time.time()
                
                # load the robot and the controller
                robot = Robot(config_path=config_path,
                            device=device)
                
                print('Time to initialize the robot and the controller:', time.time() - tnow)

                # visualize the scene
                if visualize_method:
                    pose = robot.nerf.pipeline.datamanager.eval_dataset.cameras.camera_to_worlds[0].to(device)
                    
                    depth, color = robot.render(pose.to(device))
                    h = d - depth.min().unsqueeze(0)
                    color = cv2.rectangle(color.to(device).detach().cpu().numpy(), (110, 10), (115, 58), (0, 0, 0), 1)
                    color = cv2.rectangle(color, (111, min(34-int(24*h/max_h), 34)), (114, max(34-int(24*h/max_h), 34)), (0, 0, 1), -1)
                    #videoWriter.write(np.hstack([cv2.normalize(depth.unsqueeze(-1).repeat(1,1,3).to(device).detach().cpu().numpy(),
                    #dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8), (color[:, :, [2,1,0]]*255).astype(np.uint8).clip(0,255)]))
                    cv2.namedWindow("Safety Filter", cv2.WINDOW_KEEPRATIO)
                    cv2.imshow('Safety Filter', np.hstack([cv2.normalize(depth.unsqueeze(-1).repeat(1,1,3).to(device).detach().cpu().numpy(),
                    dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX), color[:, :, [2,1,0]]]))
        
            
                ### Create configurations

                # For flightroom
                # x = torch.tensor([0.0, 0.1, 0.05, 0.0, 0.0, 0.0], device=device).to(torch.float32)
                # x = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=device).to(torch.float32)

                # x = torch.tensor([0.0, 0.0, 0.0, 0.0, 0n.0, 0.0], device=device).to(torch.float32)
                # xf = torch.tensor([0.5, 0.09, -0.04, 0.0, 0.0, 0.0], device=device).to(torch.float32)

                # For old union
                # x = torch.tensor([-0.1, 0.47, -0.17, 0.0, 0.0, 0.0], device=device).to(torch.float32)
                # xf = torch.tensor([0.35, -0.2, -0.14, 0.0, 0.0, 0.0], device=device).to(torch.float32)

                # x = torch.tensor([0.19, 0.47, -0.17, 0.0, 0.0, 0.0], device=device).to(torch.float32)
                # xf = torch.tensor([-0.28, -0.2, -0.14, 0.0, 0.0, 0.0], device=device).to(torch.float32)

                # initial and goal poses
                x0 = np.stack([radius_config*np.cos(t), radius_config*np.sin(t), radius_z * np.sin(t_z)], axis=-1)     # starting positions
                x0 = x0 + mean_config

                xf = np.stack([radius_config*np.cos(t + np.pi), radius_config*np.sin(t + np.pi), radius_z * np.sin(t_z + np.pi)], axis=-1)     # goal positions
                xf = xf + mean_config

                # Run simulation
                total_data = []
                
                for trial, (start, goal) in enumerate(zip(x0, xf)):
                    x = torch.tensor(start).to(device).to(torch.float32)
                    x = torch.cat([x, torch.zeros(3).to(device).to(torch.float32)])
                    goal = torch.tensor(goal).to(device).to(torch.float32)
                    goal = torch.cat([goal, torch.zeros(3).to(device).to(torch.float32)])
                    traj = [x]
                    times = [0]
                    u_values = []
                    u_des_values = []
                    safety = []
                    sucess = []

                    total_time = []
                    
                    for i in tqdm(range(n_steps), desc=f"Simulating trajectory {trial}"):

                        vel_des = 5.0*(goal[:3] - x[:3])
                        vel_des = torch.clamp(vel_des, -0.1, 0.1)

                        # add d term
                        vel_des = vel_des + 1.0*(goal[3:] - x[3:])
                        # cap between -0.1 and 0.1
                        u_des = 1.0*(vel_des - x[3:])
                        # cap between -0.1 and 0.1
                        u_des = torch.clamp(u_des, -0.1, 0.1)

                        tnow = time.time()
                        torch.cuda.synchronize()
                        
                        # current pose (position state in SE(3))
                        pose = state_to_pose(x[:3]).squeeze()
                        
                        # update the robot's velocity
                        robot.v = x[3:]
                        
                        # render the depth image
                        try:
                            depth, color = robot.render(pose.to(device))
                        except Exception as excp:
                            print(excp)
                        
                        # compute the 'safe' control input
                        u, h, is_safe = find_safe_action(robot,
                                                        pos_vel_state=x, 
                                                        h=d - depth.min() + beta * torch.norm(robot.v, p=2), 
                                                        intended_action=u_des,
                        )
                                                      
                        # print(f'u: {u}, u_des: {u_des}')
                        
                        # extract the control inputs for the position states
                        u = u[:3]
                        
                        # timing
                        torch.cuda.synchronize()
                        total_time.append(time.time() - tnow)

                        # integrate the dynamics
                        x_ = x
                        x = double_integrator_dynamics(x, u)*dt + x

                        traj.append(x)
                        times.append((i+1) * dt)
                        u_values.append(u.cpu().numpy())
                        u_des_values.append(u_des.cpu().numpy())

                        # record some stuff
                        print(f'State error: {torch.norm(x - goal)}')

                        # It's gotten stuck
                        if torch.norm(x - x_) < 0.0001:
                            if torch.norm(x - goal) < 0.001:
                                print("Reached Goal")
                                sucess.append(True)
                            else:
                                sucess.append(False)
                            break
                        
                        # Failure
                        if not is_safe:
                            sucess.append(False)
                            break
                            
                    if i >= n_steps - 1:
                        sucess.append(True)

                    traj = torch.stack(traj)
                    u_values = np.array(u_values)
                    u_des_values = np.array(u_des_values)

                    data = {
                        'traj': traj.cpu().numpy().tolist(),
                        'u_out': u_values.tolist(),
                        'u_des': u_des_values.tolist(),
                        'time_step': times,
                        # 'safety': safety,
                        'sucess': sucess,
                        'total_time': total_time,
                    }

                    total_data.append(data)
            finally:
                # Save trajectory
                data = {
                    'scene': scene_name,
                    'method': method,
                    'alpha': alpha,
                    'beta': beta,
                    'radius': radius,
                    'dt': dt,
                    'total_data': total_data,
                }

                with open(f'{data_output_path}/{scene_name}_{method}.json', 'w') as f:
                    json.dump(data, f, indent=4)
#%%