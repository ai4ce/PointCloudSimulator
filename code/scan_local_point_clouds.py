import os
import sys
import argparse
import pickle

import numpy as np
from scipy.io import loadmat
import torch
from torch.utils.data import Dataset

from DPS import BaseDPS
from open3d import PointCloud,Vector3dVector,write_point_cloud

parser = argparse.ArgumentParser()
parser.add_argument('-d','--data_dir',type=str,default='../data/map_pose_traj/',help='dataset path')
parser.add_argument('-v','--env',type=str,default='v1',help='environment name')
parser.add_argument('-o','--obs',type=int,default=256,help='number of obs')
parser.add_argument('--pose',type=str,default='pose0',help='pose namej')
#parser.add_argument('--lim',type=float,default=None,help='sense limitation')

def np_to_pcd(xyz):
    xyz = xyz.reshape(-1,3) 
    pcd = PointCloud()
    pcd.points = Vector3dVector(xyz)
    return pcd

opt = parser.parse_args()
save_dir = os.path.join(opt.data_dir,opt.env,opt.env+'_'+opt.pose)
# load map
map_file = os.path.join(opt.data_dir,opt.env,'environment.pkl')
with open(map_file,'rb') as f:
    maps = pickle.load(f)

# load pose
pose_file = os.path.join(save_dir,'gt_pose.mat')
pose_mat_file = loadmat(pose_file)
poses = pose_mat_file['pose'].astype(np.float32)
poses = torch.from_numpy(poses)

img_shape = maps.image.shape 

dps = BaseDPS(number_of_obs=opt.obs,img_shape=img_shape,fov=2*np.pi,normalized=True) 

with torch.no_grad():
    dps(maps,poses)

#dps.save_obs_global(save_dir)
#dps.save_obs_local(save_dir)
obs = dps.obs_local.cpu().detach().numpy()

print('Env: {}, Pose: {}'.format(opt.env,opt.pose))

n_obs = obs.shape[0]
for i in range(n_obs):
    xy = obs[i,:,:]
    zero = np.zeros_like(xy)[:,0:1]
    xyz = np.concatenate((xy,zero),axis=-1) # <Lx3>
    pcd = np_to_pcd(xyz) 
    
    save_name = '{:09d}.pcd'.format(i)
    save_name = os.path.join(save_dir,save_name)
    write_point_cloud(save_name,pcd)
