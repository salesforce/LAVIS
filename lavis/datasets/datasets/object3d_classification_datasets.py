"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

# Adapted from https://github.com/salesforce/ULIP/blob/48d8d00b1cdb2aee79005817a202816f1c521911/models/pointnext/PointNeXt/openpoints/dataset/modelnet/modelnet40_normal_resampled_loader.py

import os
from collections import OrderedDict
import numpy as np
from tqdm import tqdm
import torch
import copy
import random
import pickle
from PIL import Image
from lavis.processors.ulip_processors import farthest_point_sample, pc_normalize
from lavis.datasets.datasets.base_dataset import BaseDataset


class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": ann["image"],
                "caption": ann["caption"],
                "image": sample["image"],
                "pc": sample["pc"],
            }
        )

class ModelNetClassificationDataset(BaseDataset, __DisplMixin):
    """
    Dataset for ModelNet Classification.
    """
    def __init__(self, **kwargs):
        super().__init__(kwargs['vis_processor'], kwargs['text_processor'], kwargs['vis_root'], [])

        self.modalities = kwargs['modalities']
        # Setting dataset specific properties
        self.npoints = 8192
        self.use_normals = False
        self.num_category = 40
        self.process_data = True
        self.uniform = True
        self.generate_from_raw_data = False
        ann_paths = kwargs['ann_paths']

        assert 'pc_root' in kwargs, "Point cloud root needs to be provided to retrieve labels."
        self.pc_root = kwargs["pc_root"]

        # Fetching class names and IDs
        self.classnames = [line.rstrip() for line in open(ann_paths[0])]
        self.classes = dict(zip(self.classnames, range(len(self.classnames))))
        self.shape_ids = [line.rstrip() for line in open(ann_paths[-1])]
        self.shape_names = ['_'.join(x.split('_')[0:-1]) for x in self.shape_ids]

        # Setting data paths
        self.datapath = [(self.shape_names[i], os.path.join(self.pc_root, self.shape_names[i], self.shape_ids[i]) + '.txt') for i
                         in range(len(self.shape_ids))]


        # Saving path settings
        self.save_path = ann_paths[1] if self.uniform else ann_paths[0].replace('_fps', '')
        
        # Processing or loading data
        self._prepare_data()


    def _prepare_data(self):
        # Check for pre-processed data
        if self.process_data:
            if not os.path.exists(self.save_path):
                if self.generate_from_raw_data:
                    print('Processing data %s (only running in the first time)...' % self.save_path)
                    self._process_raw_data()
            else:
                print('Load processed data from %s...' % self.save_path)
                with open(self.save_path, 'rb') as f:
                    self.list_of_points, self.list_of_labels = pickle.load(f)
        else:
            print('Load processed data from %s...' % self.save_path)
            with open(self.save_path, 'rb') as f:
                self.list_of_points, self.list_of_labels = pickle.load(f)

    def _process_raw_data(self):
        self.list_of_points = [None] * len(self.datapath)
        self.list_of_labels = [None] * len(self.datapath)
        for index in tqdm(range(len(self.datapath)), total=len(self.datapath)):
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            cls = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)
                print("uniformly sampled out {} points".format(self.npoints))
            else:
                point_set = point_set[0:self.npoints, :]

            self.list_of_points[index] = point_set
            self.list_of_labels[index] = cls

        with open(self.save_path, 'wb') as f:
            pickle.dump([self.list_of_points, self.list_of_labels], f)

    def __len__(self):
        return len(self.list_of_labels)

    def _get_item(self, index):
        if self.process_data:
            point_set, label = self.list_of_points[index], self.list_of_labels[index]
        else:
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            label = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)
            
            # Uniform sampling or trimming
            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)
            else:
                point_set = point_set[0:self.npoints, :]
        if self.npoints < point_set.shape[0]:
            point_set = farthest_point_sample(point_set, self.npoints)

        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        if not self.use_normals:
            point_set = point_set[:, 0:3]
            
        return point_set, label[0]

    def __getitem__(self, index):
        points, label = self._get_item(index)
        label_name = self.classnames[int(label)]

        data =  {
                "instance_id": index,
                "sample_key": index,
                "image_id": index,
                "label": label_name
                }
        
        if 'pc' in self.modalities:
            pt_idxs = np.arange(0, points.shape[0])
            np.random.shuffle(pt_idxs)
            current_points = points[pt_idxs].copy()
            current_points = torch.from_numpy(current_points).float()
            data['pc'] = current_points
        if any([k in self.modalities for k in ['images', 'image']]):
            img = Image.open(os.path.join(self.vis_root,f"{index}.jpeg" ))
            data['image'] = self.vis_processor(img)

        return data
