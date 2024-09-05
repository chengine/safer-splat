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


# with open('device.txt', encoding='utf-8') as file:
#      device=file.read()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
time_step = 0.1
mu = 0
alpha = 0.5
d = 0.1
u = torch.tensor([[0,0,1]]).t().float().to(device)
max_h = 1.5
intend = sigma = 1
batch_size = 10

def state_to_pose(state):
    for i in range(state.shape[0]):
        rm = torch.from_numpy(R.from_euler('xyz', state[i][3:].cpu(), degrees=True).as_matrix()).to(device)
        pose = torch.cat((torch.cat((rm, state[i][:3].unsqueeze(-1)), 1), torch.tensor([[0,0,0,1]]).to(device)), 0).unsqueeze(0)
        if i == 0:
            batch_pose = pose
        else:
            batch_pose = torch.cat((batch_pose, pose), dim=0)
    return batch_pose

def update_dynamics(state, action):
    return state + time_step * action


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

    def predict_observation(self, pose):
        # update the pose of the camera
        self.camera.camera_to_worlds = pose[None, :3, :] if len(pose.shape) == 2 else pose[:, :3, :]
        
        # render the observations
        outputs = self.nerf.render(cameras=self.camera)
        
        # extract the depth
        depth = outputs['depth']
        
        return depth
    
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
        
        return outputs['depth'], outputs['rgb']

def find_safe_action(robot, pose, h, intended_action, direction):
    state = torch.cat((pose[:3, -1].to(device), torch.from_numpy(R.from_matrix(pose[:3, :3].cpu()).as_euler('xyz', degrees=True)).to(device)), dim=0)
    orient_action = torch.zeros(6).to(device)
    if direction in ['up', 'down']:
        unit = intended_action * torch.mm(pose[:3, :3].float(), u).squeeze().to(device)
        orient_action[0] = unit[0]
        orient_action[1] = unit[1]
        orient_action[2] = unit[2]
    elif direction in ['left', 'right']:
        orient_action[5] = intended_action * 10
    new_state = update_dynamics(state, orient_action).unsqueeze(0)
    new_pose = state_to_pose(new_state)
    new_h = d - robot.predict_observation(new_pose).min().unsqueeze(0)
    best_action = torch.zeros(6).to(device)
    if new_h <= alpha * h:
        print('Intended action {} is safe'.format(orient_action))
        print('intervention = 0')
        return orient_action, new_h, True
    while True:
        batch_action = torch.zeros((batch_size, 6)).to(device)
        if direction in {'up', 'down'}:
            for j in range(batch_size):
                value = np.random.normal(mu, sigma)
                while value * intended_action > 0 and abs(value) >= abs(intended_action):
                    value = np.random.normal(mu, sigma)
                unit = value * torch.mm(pose[:3, :3].float(), u).squeeze().to(device)
                batch_action[j][0] = unit[0]
                batch_action[j][1] = unit[1]
                batch_action[j][2] = unit[2]
        else:
            for j in range(batch_size):
                batch_action[j][5] = np.random.normal(mu, sigma) * 10
        batch_new_state = update_dynamics(state, batch_action)
        batch_new_pose = state_to_pose(batch_new_state)
        batch_new_h = d - robot.predict_observation(batch_new_pose).min(dim=-1)[0].min(dim=-1)[0]
        for j in range(batch_size):
            if batch_new_h[j] <= alpha * h:
                if torch.norm(best_action, p=2) == 0 or torch.norm(batch_action[j] - orient_action, p=2) < torch.norm(best_action - orient_action, p=2):
                    best_action = batch_action[j]
                    new_h = batch_new_h[j]
        if torch.norm(best_action, p=2) > 0:
            print('Intended action {} is unsafe, a recommended substitute is {}'.format(orient_action, best_action))
            print('intervention =', float(torch.norm(orient_action - best_action, p=2)))
            return best_action, new_h, False
    #print('Fail to find a safe action')
    #return best_action, h, False


if __name__ == '__main__':
    # config path
    config_path = Path(f"{os.path.expanduser('~/Research/Gaussian_Splatting/outputs/nerfstudio/stonehenge/nerfacto/2024-09-04_130335/config.yml')}")

    # robot
    robot = Robot(config_path=config_path,
                  device=device)
    
    # other parameters
    fps = 10
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    videoWriter = cv2.VideoWriter('video.mp4',fourcc,fps,(240,68))
    
    # initial pose
    
    # pose = state_to_pose(torch.tensor([-2, 0.2,  0.5, 90, 0, 115]).unsqueeze(0).to(device)).squeeze()
    
    pose = robot.nerf.pipeline.datamanager.eval_dataset.cameras.camera_to_worlds[0].to(device)
    
    depth, color = robot.render(pose.to(device))
    h = d - depth.min()
    color = cv2.rectangle(color.to(device).detach().cpu().numpy(), (1100, 100), (1150, 580), (0, 0, 0), 10)
    color = cv2.rectangle(color, (1110,  min(340-int(240*h/max_h), 340)), (1140, max(340-int(240*h/max_h), 340)), (0, 0, 1), -1)
    color = cv2.putText(color, 'CBF h', (1025, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.75, (0, 0, 0), 5)
    plt.imshow(np.hstack([cv2.normalize(depth.repeat(1,1,3).to(device).detach().cpu().numpy(),
            dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX), color]))
    print('min_depth = {}'.format(depth.min()))
    
    stop_next = -1
    frame = 0
    while frame < 150:
        direction = 0
        if direction == 0 : #up
            frame += 1
            print(frame)
            intended_action = -intend
            action, h, is_safe = find_safe_action(robot, pose, d - depth.min(), intended_action, 'up')
            state = torch.cat((pose[:3, -1].to(device), torch.from_numpy(R.from_matrix(pose[:3, :3].cpu()).as_euler('xyz', degrees=True)).to(device)), dim=0).to(device)
            state = update_dynamics(state, action)
            pose = state_to_pose(state.unsqueeze(0)).squeeze()
            depth, color = robot.render(pose.to(device))
            color = cv2.rectangle(color.to(device).detach().cpu().numpy(), (1100, 100), (1150, 580), (0, 0, 0), 10)
            if is_safe:
                color = cv2.rectangle(color, (1110, min(340-int(240*h/max_h), 340)), (1140, max(340-int(240*h/max_h), 340)), (0, 0, 1), -1)
            else:
                color = cv2.rectangle(color, (1110, min(340-int(240*h/max_h), 340)), (1140, max(340-int(240*h/max_h), 340)), (1, 0, 0), -1)
            color = cv2.putText(color, 'CBF h', (1025, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.75, (0, 0, 0), 5)
            # plt.imshow(np.hstack([cv2.normalize(depth.unsqueeze(-1).repeat(1,1,3).to(device).detach().cpu().numpy(),
            # dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX), color]))
            # plt.xticks([])
            # plt.yticks([])
            # plt.savefig('t_{}.png'.format(frame), bbox_inches='tight', pad_inches=0.0)
            videoWriter.write(np.hstack([cv2.normalize(depth.repeat(1,1,3).to(device).detach().cpu().numpy(), dst=None, 
            alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8), (color[:, :, [2,1,0]]*255).astype(np.uint8).clip(0,255)]))
            print('min_depth = {}'.format(depth.min()))
        elif direction == 1 and stop_next != 1:  # down
            frame += 1
            print(frame)
            stop_next = -1
            action, h, is_safe = find_safe_action(robot, pose, d - depth.min(), intend, 'down')
            state = torch.cat((pose[:3, -1].to(device), torch.from_numpy(R.from_matrix(pose[:3, :3].cpu()).as_euler('xyz', degrees=True)).to(device)), dim=0).to(device)
            state = update_dynamics(state, action)
            pose = state_to_pose(state.unsqueeze(0)).squeeze()
            depth, color = robot.render(pose.to(device))
            color = cv2.rectangle(color.to(device).detach().cpu().numpy(), (110, 10), (115, 58), (0, 0, 0), 1)
            if is_safe:
                color = cv2.rectangle(color, (111, min(34-int(24*h/max_h), 34)), (114, max(34-int(24*h/max_h), 34)), (0, 0, 1), -1)
            else:
                stop_next = 1
                color = cv2.rectangle(color, (111, min(34-int(24*h/max_h), 34)), (114, max(34-int(24*h/max_h), 34)), (1, 0, 0), -1)
            # plt.imshow(np.hstack([cv2.normalize(depth.unsqueeze(-1).repeat(1,1,3).to(device).detach().cpu().numpy(),
            # dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX), color]))
            # plt.xticks([])
            # plt.yticks([])
            # plt.savefig('t_{}.png'.format(frame), bbox_inches='tight', pad_inches=0.0)
            videoWriter.write(np.hstack([cv2.normalize(depth.repeat(1,1,3).to(device).detach().cpu().numpy(), dst=None, 
            alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8), (color[:, :, [2,1,0]]*255).astype(np.uint8).clip(0,255)]))
            print('min_depth = {}'.format(depth.min()))
        elif direction == 2 and stop_next != 2:  # left
            frame += 1
            print(frame)
            stop_next = -1
            action, h, is_safe = find_safe_action(robot, pose, d - depth.min(), intend, 'left')
            state = torch.cat((pose[:3, -1].to(device), torch.from_numpy(R.from_matrix(pose[:3, :3].cpu()).as_euler('xyz', degrees=True)).to(device)), dim=0).to(device)
            state = update_dynamics(state, action)
            pose = state_to_pose(state.unsqueeze(0)).squeeze()
            depth, color = robot.render(pose.to(device))
            color = cv2.rectangle(color.to(device).detach().cpu().numpy(), (110, 10), (115, 58), (0, 0, 0), 1)
            if is_safe:
                color = cv2.rectangle(color, (111, min(34-int(24*h/max_h), 34)), (114, max(34-int(24*h/max_h), 34)), (0, 0, 1), -1)
            else:
                stop_next = 2
                color = cv2.rectangle(color, (111, min(34-int(24*h/max_h), 34)), (114, max(34-int(24*h/max_h), 34)), (1, 0, 0), -1)
            # plt.imshow(np.hstack([cv2.normalize(depth.unsqueeze(-1).repeat(1,1,3).to(device).detach().cpu().numpy(),
            # dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX), color]))
            # plt.xticks([])
            # plt.yticks([])
            # plt.savefig('t_{}.png'.format(frame), bbox_inches='tight', pad_inches=0.0)
            videoWriter.write(np.hstack([cv2.normalize(depth.repeat(1,1,3).to(device).detach().cpu().numpy(), dst=None, 
            alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8), (color[:, :, [2,1,0]]*255).astype(np.uint8).clip(0,255)]))
            print('min_depth = {}'.format(depth.min()))
        elif direction == 3 and stop_next != 3:  # right
            frame += 1
            print(frame)
            stop_next = -1
            action, h, is_safe = find_safe_action(robot, pose, d - depth.min(), -intend, 'right')
            state = torch.cat((pose[:3, -1].to(device), torch.from_numpy(R.from_matrix(pose[:3, :3].cpu()).as_euler('xyz', degrees=True)).to(device)), dim=0).to(device)
            state = update_dynamics(state, action)
            pose = state_to_pose(state.unsqueeze(0)).squeeze()
            depth, color = robot.render(pose.to(device))
            color = cv2.rectangle(color.to(device).detach().cpu().numpy(), (110, 10), (115, 58), (0, 0, 0), 1)
            if is_safe:
                color = cv2.rectangle(color, (111, min(34-int(24*h/max_h), 34)), (114, max(34-int(24*h/max_h), 34)), (0, 0, 1), -1)
            else:
                stop_next = 3
                color = cv2.rectangle(color, (111, min(34-int(24*h/max_h), 34)), (114, max(34-int(24*h/max_h), 34)), (1, 0, 0), -1)
            # plt.imshow(np.hstack([cv2.normalize(depth.unsqueeze(-1).repeat(1,1,3).to(device).detach().cpu().numpy(),
            # dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX), color]))
            # plt.xticks([])
            # plt.yticks([])
            # plt.savefig('t_{}.png'.format(frame), bbox_inches='tight', pad_inches=0.0)
            videoWriter.write(np.hstack([cv2.normalize(depth.repeat(1,1,3).to(device).detach().cpu().numpy(), dst=None, 
            alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8), (color[:, :, [2,1,0]]*255).astype(np.uint8).clip(0,255)]))
            print('min_depth = {}'.format(depth.min()))

    videoWriter.release()

# %%
