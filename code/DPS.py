import os
from copy import deepcopy
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import torch
import torch.nn as nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
class BaseDPS(nn.Module):
    def __init__(self,number_of_obs,img_shape,fov,normalized,sense_lim=None):
        super(BaseDPS, self).__init__()
        self.number_of_obs = number_of_obs
        self.fov = fov
        self.img_shape = img_shape
        #self.n_rot = n_rot
        self.normalized = normalized
        self.sense_lim = sense_lim


    def rays(self, poses):
        '''
        compute rays for each poses
        :param poses: <Bx3> x,y,theta
        :return: <BxLx3> L is #obs
        '''
        L = self.number_of_obs
        B = poses.size(0)

        angles = torch.from_numpy(np.arange(L,dtype=np.float32) * self.fov / L).to(poses.device) # anlges of each ray

        angles = angles.unsqueeze(0).expand(B,-1) + poses[:, -1].unsqueeze(-1) #<BxL>
        angles = angles.unsqueeze(-1)
        p0 = poses[:,:2].unsqueeze(1).expand(-1,L,-1) #<BxLx2>
        ret = torch.cat((p0,angles),dim=-1) #<BxLx3>

        return ret

    def find_intersection(self,map,pose_ray):
        L = self.number_of_obs
        B = pose_ray.shape[0]
        N = map.shape[0]

        p,theta = pose_ray[:,:,:2],pose_ray[:,:,2:] # p: <BxLx2> theta: <BxLx1>
        #------------- Ray: p-----------------------#
        x1 = pose_ray[:,:,0].unsqueeze(-1).expand(-1,-1,N)
        y1 = pose_ray[:,:,1].unsqueeze(-1).expand(-1,-1,N) # <BxLxN>
        rx = torch.cos(theta).expand(-1,-1,N)
        ry = torch.sin(theta).expand(-1,-1,N)

        #-------Line_segs: q------#
        tmp_map = map.clone().expand(B,-1,-1) #<BxNx4>
        x3 = tmp_map[:,:,0].unsqueeze(1).expand(-1,L,-1)
        y3 = tmp_map[:,:,1].unsqueeze(1).expand(-1,L,-1)
        x4 = tmp_map[:,:,2].unsqueeze(1).expand(-1,L,-1)
        y4 = tmp_map[:,:,3].unsqueeze(1).expand(-1,L,-1) # x... and y... <BxLxN>
        
        sx = x4-x3
        sy = y4-y3
        # p-q
        x1_x3 = x1-x3
        y1_y3 = y1-y3

        # cross(s,(p-q))
        numerator_a = sx * y1_y3 - sy * x1_x3
        # cross(r,(p-q))
        numerator_b = rx* y1_y3 - ry * x1_x3
        # cross(r,s)
        rxs = sy * rx- sx * ry

        # find parallel lines
        is_parallel = torch.abs(rxs)< 1e-4
        zero = torch.tensor(0.).to(rxs.device)
        rxs = torch.where(is_parallel,zero,rxs)
        numerator_a = torch.where(is_parallel,zero,numerator_a)
        numerator_b = torch.where(is_parallel,zero,numerator_b)

        u_a = numerator_a /rxs
        u_b = numerator_b /rxs


        mask_b = np.logical_and(u_b>=0,u_b<=1)
        mask = np.logical_and(mask_b,u_a>=0)

        intersect_x = x1+rx*u_a # <BxLxN>
        intersect_y = y1+ry*u_a

        mask = 1 - mask.type(torch.float)
        mask = mask.to(pose_ray.device)
        u_a = u_a + torch.ones_like(u_a)*10000000*mask #u_a[~mask] = np.inf

        #idx = np.argmin(u_a,axis=-1) # <BxL>
        idx = np.nanargmin(u_a.cpu().detach().numpy(),axis=-1) # <BxL>


        batch_ind = torch.arange(0,B,dtype=torch.long).repeat(L).reshape(L,B).t().reshape(-1)
        obs_ind = torch.arange(0,L,dtype=torch.long).repeat(B)

        intersect_x = intersect_x[batch_ind,obs_ind,idx[batch_ind,obs_ind]].reshape(B,L)
        intersect_y = intersect_y[batch_ind,obs_ind,idx[batch_ind,obs_ind]].reshape(B,L)

        obs = torch.stack((intersect_x,intersect_y),dim=2)
        return obs


    def to_sensor_frame(self,pose,obs_global):
        """
        transform obs global coordinate to sensor coordinate frame
        :param pose: <Bx3>
        :param obs_global: <BxLx2>
        :return obs_local: <BxLx2>
        """

        # row-based matrix product
        L = obs_global.shape[1]
        c0,theta0 = pose[:,0:2],pose[:,2] # c0 is the loc of sensor in global coord. frame c0: <Bx2>
        c0 = c0.unsqueeze(1).expand(-1,L,-1)

        #tmp = theta0.detach().numpy()
        #cos = torch.from_numpy(np.cos(tmp)).unsqueeze(-1).unsqueeze(-1)
        #sin = torch.from_numpy(np.sin(tmp)).unsqueeze(-1).unsqueeze(-1)
        cos = torch.cos(theta0).unsqueeze(-1).unsqueeze(-1)
        sin = torch.sin(theta0).unsqueeze(-1).unsqueeze(-1)
        R = torch.cat((cos,-sin,sin,cos),dim=1).reshape(-1,2,2)

        obs_local = torch.bmm(obs_global-c0,R)

        return obs_local


    def sense(self, map, pose):
        '''
        given a map, B poses, return B observations
        :param map: <Nx4>
        :param pose: <Bx3>
        :return: <BxLx2>, in sensor coordinate frame
        '''

        rays = self.rays(pose) #<BxLx4>
        rays = rays.to(pose.device)
        obs_global = self.find_intersection(map,rays)  
        return obs_global

    def _normalize_map_pose(self):
        self.map.line_seg = self._normalize(self.map.line_seg,self.img_shape[1])
        self.pose[:,:2] = self._normalize(self.pose[:,:2],self.img_shape[1])
        if hasattr(self.map,'pc'):
            self.map.pc[:,0:2] = self._normalize(self.map.pc[:,0:2],self.img_shape[1])

    def _denormalize_map_pose(self):
        self.map.line_seg = self._denormalize(self.map.line_seg,self.img_shape[1])
        self.pose_est[:,:2] = self._denormalize(self.pose_est[:,:2],self.img_shape[1])
        self.obs_global = self._denormalize(self.obs_global,self.img_shape[1])
        self.pose[:,:2] = self._denormalize(self.pose[:,:2],self.img_shape[1])

    def forward(self,map,pose):
        map,pose = deepcopy(map),deepcopy(pose)
        bs = pose.size(0)
        self.map,self.pose = map,pose
        if self.normalized:
            self._normalize_map_pose()

        self.obs_global = self.sense(map.line_seg, pose) # <BxLx2>
        self.obs_local = self.to_sensor_frame(pose,self.obs_global)

        if self.sense_lim is not None:
            obs_dist =  torch.sqrt(torch.sum(self.obs_local**2,-1)) #<BxL>
            if self.normalized:
                obs_dist = obs_dist/2*self.img_shape[1]

            self.out_of_range = obs_dist > self.sense_lim
            if  self.out_of_range.any(): # if any local obs are out of range, otherwise do nothing
                factor = torch.unsqueeze(self.sense_lim/obs_dist[self.out_of_range],-1)
                self.obs_local[self.out_of_range] *= factor
        else:
            self.out_of_range = torch.zeros( (bs,self.number_of_obs) , dtype=torch.uint8 )
            self.out_of_range = self.out_of_range.to(self.obs_local.device)

    def _normalize(self,x,s): return 2*(x-s/2)/s
    def _denormalize(self,x,s): return (x+1)*s/2

    def save_global_map(self,save_dir,point_color='k',pose_color='r'):
        obs_global = self.obs_global.cpu().detach().numpy()
        pose = self.pose.cpu().detach().numpy()
        bs = obs_global.shape[0]
        save_name = os.path.join(save_dir,'global_map')
        for idx in range(bs):
            plt.plot(obs_global[idx,:,0],obs_global[idx,:,1],'.',color=point_color)
        plt.plot(pose[:,0],pose[:,1],'.-',color=pose_color)
        plt.savefig(save_name)
        plt.close()

    def save_obs_global(self,save_dir,color='k'):
        obs_global = self.obs_global.cpu().detach().numpy()
        bs = obs_global.shape[0]
        for idx in range(bs):
            save_name = os.path.join(save_dir,'global_obs_'+str(idx))
            gca = plt.gca()
            plt.plot(obs_global[idx,:,0],obs_global[idx,:,1],'.',color=color)
            gca.axes.xaxis.set_ticklabels([])
            gca.axes.yaxis.set_ticklabels([])
            plt.savefig(save_name)
            plt.close()

    def save_obs_local(self,save_dir,color='k'):
        obs_local = self.obs_local.cpu().detach().numpy()
        bs = obs_local.shape[0]
        for idx in range(bs):
            save_name = os.path.join(save_dir,'local_obs_'+str(idx))
            gca = plt.gca()
            plt.plot(obs_local[idx,:,0],obs_local[idx,:,1],'.',color=color)
            gca.axes.xaxis.set_ticklabels([])
            gca.axes.yaxis.set_ticklabels([])
            plt.savefig(save_name)
            plt.close()
