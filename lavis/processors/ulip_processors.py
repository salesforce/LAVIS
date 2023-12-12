"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

"""
Adapted from ULIP codebase: https://github.com/salesforce/ULIP
"""

from lavis.common.registry import registry
from lavis.processors.blip_processors import BlipImageBaseProcessor
from omegaconf import OmegaConf
import torchvision.transforms as transforms
from lavis.models.ulip_models.utils.io import IO

import numpy as np
from PIL import Image
import torch


def pc_norm(pc):
    """ pc: NxC, return NxC """
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

def random_sample(permutation, pc, num):
    np.random.shuffle(permutation)
    pc = pc[permutation[:num]]
    return pc

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data

def random_point_dropout(batch_pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 '''
    for b in range(batch_pc.shape[0]):
        dropout_ratio =  np.random.random()*max_dropout_ratio # 0~0.875
        drop_idx = np.where(np.random.random((batch_pc.shape[1]))<=dropout_ratio)[0]
        if len(drop_idx)>0:
            batch_pc[b,drop_idx,:] = batch_pc[b,0,:] # set to the first point
    return batch_pc

def random_scale_point_cloud(batch_data, scale_low=0.8, scale_high=1.25):
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            BxNx3 array, original batch of point clouds
        Return:
            BxNx3 array, scaled batch of point clouds
    """
    B, N, C = batch_data.shape
    scales = np.random.uniform(scale_low, scale_high, B)
    for batch_index in range(B):
        batch_data[batch_index,:,:] *= scales[batch_index]
    return batch_data

def shift_point_cloud(batch_data, shift_range=0.1):
    """ Randomly shift point cloud. Shift is per point cloud.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, shifted batch of point clouds
    """
    B, N, C = batch_data.shape
    shifts = np.random.uniform(-shift_range, shift_range, (B,3))
    for batch_index in range(B):
        batch_data[batch_index,:,:] += shifts[batch_index,:]
    return batch_data

def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data += batch_data
    return jittered_data

def rotate_perturbation_point_cloud(batch_data, angle_sigma=0.06, angle_clip=0.18):
    """ Randomly perturb the point clouds by small rotations
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        angles = np.clip(angle_sigma*np.random.randn(3), -angle_clip, angle_clip)
        Rx = np.array([[1,0,0],
                       [0,np.cos(angles[0]),-np.sin(angles[0])],
                       [0,np.sin(angles[0]),np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]),0,np.sin(angles[1])],
                       [0,1,0],
                       [-np.sin(angles[1]),0,np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]),-np.sin(angles[2]),0],
                       [np.sin(angles[2]),np.cos(angles[2]),0],
                       [0,0,1]])
        R = np.dot(Rz, np.dot(Ry,Rx))
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), R)
    return rotated_data


@registry.register_processor("ulip_pc")
class ULIPPCProcessor(BlipImageBaseProcessor):
    def __init__(
        self, 
        npoints=8192,
        augment=False,
        uniform=True,
        ssl=False,
        oversample=False,
        use_height=False, 
    ):

        super().__init__()

        self.npoints=npoints
        self.augment=augment
        self.uniform=uniform
        self.ssl=ssl
        self.oversample=oversample
        self.use_height=use_height
        self.permutation = np.arange(self.npoints)


    def __call__(self, pc_data_path):
        if isinstance(pc_data_path, np.ndarray):
            pc_data = pc_data_path
        else:
            try:
                pc_data = np.load(pc_data_path, allow_pickle=True)['arr_0'].astype(np.float32)
            except:
                pc_data = IO.get(pc_data_path).astype(np.float32)
        data = pc_norm(pc_data)

        if self.uniform and self.npoints < data.shape[0]:
            data = farthest_point_sample(data, self.npoints)
        else:
            data = random_sample(self.permutation, data, self.npoints)

        if self.augment:
            data = random_point_dropout(data[None, ...])
            data = random_scale_point_cloud(data)
            data = shift_point_cloud(data)
            data = rotate_perturbation_point_cloud(data)
            data = rotate_point_cloud(data)
            data = data.squeeze()

        if self.ssl:
            data_for_aug = data[:]
            data_aug_1 = random_point_dropout(data_for_aug[None, ...])
            data_aug_1 = random_scale_point_cloud(data_aug_1, scale_low=0.5, scale_high=1.5)
            data_aug_1 = shift_point_cloud(data_aug_1, shift_range=0.4)
            data_aug_1 = rotate_perturbation_point_cloud(data_aug_1, angle_sigma=0.1, angle_clip=0.3)
            data_aug_1 = rotate_point_cloud(data_aug_1)
            data_aug_1 = data_aug_1.squeeze()

            data_aug_2 = random_point_dropout(data_for_aug[None, ...])
            data_aug_2 = random_scale_point_cloud(data_aug_2, scale_low=0.5, scale_high=1.5)
            data_aug_2 = shift_point_cloud(data_aug_2, shift_range=0.4)
            data_aug_2 = rotate_perturbation_point_cloud(data_aug_2, angle_sigma=0.1, angle_clip=0.3)
            data_aug_2 = rotate_point_cloud(data_aug_2)
            data_aug_2 = data_aug_2.squeeze()

        if self.use_height:
            self.gravity_dim = 1
            height_array = data[:, self.gravity_dim:self.gravity_dim + 1] - data[:,
                                                                       self.gravity_dim:self.gravity_dim + 1].min()
            data = np.concatenate((data, height_array), axis=1)
            data = torch.from_numpy(data).float()
        else:
            data = torch.from_numpy(data).float()
        
        if self.ssl:
            return {"data": data, "data_aug_1": data_aug_1, "data_aug_2": data_aug_2}
        else:
            return data
        
    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        npoints= cfg.get('npoints', 8192)
        augment= cfg.get('augment',False)
        uniform= cfg.get('uniform',True)
        ssl= cfg.get('ssl',False)
        oversample= cfg.get('oversample',False)
        use_height= cfg.get('use_height',False)

        return cls(
            npoints=npoints,
            augment=augment,
            uniform=uniform,
            ssl=ssl,
            oversample=oversample,
            use_height=use_height, 
        )