"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os, sys
from collections import OrderedDict
import random

from lavis.datasets.datasets.base_dataset import BaseDataset
from lavis.common.utils import is_serializable

from PIL import Image
import numpy as np
from tqdm import tqdm
import json
import torch
import copy

class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]
        serializable_sample_keys = [k for k,v in sample.items() if is_serializable(v)]
        serializable_ann_keys = [k for k,v in ann.items() if is_serializable(v)]
        display = {k:sample[k] for k in serializable_sample_keys}
        display.update({k:ann[k] for k in serializable_ann_keys})

        return OrderedDict(
            display
        )



class Object3dCaptionDataset(BaseDataset, __DisplMixin):
    def __init__(self, **kwargs):
        super().__init__(kwargs['vis_processor'], kwargs['text_processor'], kwargs['vis_root'], kwargs['ann_paths'])

        self.modalities = kwargs['modalities']
        self.npoints = 8192
        self.sample_points_num = self.npoints

        for modality in self.modalities:
            if 'image' in modality:
                setattr(self, f"existing_{modality}_annotation",getattr(self, f'get_existing_{modality}_annotations')())
                continue
            setattr(self, f"{modality}_root", kwargs[f"{modality}_root"])
            setattr(self, f"{modality}_processor", kwargs[f"{modality}_processor"])
            setattr(self, f"existing_{modality}_annotation",getattr(self, f'get_existing_{modality}_annotations')())
        self.sample_ids = set.intersection(*[set(getattr(self, f"existing_{modality}_annotation")) for modality in self.modalities])
        self.annotation = [ann for ann in self.annotation if ann['sample_id'] in self.sample_ids]
    
    def get_existing_depth_annotations(self):
        return os.listdir(self.depth_root)
    
    def get_existing_images_annotations(self):
        return os.listdir(self.vis_root)
    
    def get_existing_pc_annotations(self):
        raise NotImplementedError("Subclasses should implement this!")

    def get_pc_path(self, sample_key):
        raise NotImplementedError("Subclasses should implement this!")
    
    def get_images_path(self, sample_key):
        raise NotImplementedError("Subclasses should implement this!")
    
    def get_depth_path(self, sample_key):
        raise NotImplementedError("Subclasses should implement this!")

    def __getitem__(self, index):
        ann = copy.deepcopy(self.annotation[index])
        ann['captions'] = ann['data']
        del ann['data']
        
        for modality in self.modalities:
            ann[f"{modality}_path"] = getattr(self, f"get_{modality}_path")(ann['sample_id'])
            if type(ann[f"{modality}_path"]) == list: # select from image views
                ann[f"{modality}_path"] = random.choice(ann[f"{modality}_path"])
            if 'image' in modality:
                ann['image'] = self.vis_processor(Image.open(ann[f"images_path"]))
            else:
                ann[modality] = getattr(self, f"{modality}_processor")(ann[f"{modality}_path"]).to(torch.float32)
        return ann
    
    def __len__(self):
        return len(self.annotation)
    
    def _build_templates(self, templates_path):
        # use captions not templates
        if templates_path is None:
            self.templates = None
        else:
            with open(templates_path) as f:
                self.templates = json.load(f)


class ObjaverseCaptionDataset(Object3dCaptionDataset, __DisplMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def get_existing_images_annotations(self):
       return [f.split('_')[0] for f in os.listdir(os.path.join(self.vis_root, f'compressed_imgs_view{0}/Cap3D_imgs_view{0}/'))]
    
    def get_existing_pc_annotations(self):
        return list(set(os.listdir(self.pc_root)).intersection(set(ann['sample_id'] for ann in self.annotation)))

    def get_pc_path(self, sample_key):
        return os.path.join(self.pc_root, sample_key, '{}_{}.npz'.format(sample_key, self.npoints))
       
    def get_images_path(self, sample_key):
        # data downloaded from: https://huggingface.co/datasets/tiange/Cap3D/tree/main/RenderedImage_zips
        return [os.path.join(self.vis_root, f'compressed_imgs_view{i}/Cap3D_imgs_view{i}/', sample_key+f'_{i}.jpeg') for i in range(8)]
        
    def __getitem__(self, index):
        ann = super().__getitem__(index)
        ann['text_input'] = self.text_processor(random.choice(ann['captions']))
        return ann

class ObjaverseCaptionInstructDataset(ObjaverseCaptionDataset):
    def __getitem__(self, index):
        data = super().__getitem__(index)
        if data != None:
            data['text_output'] = data["text_input"]
            data['text_input'] = self.text_processor("")
        return data


class ObjaverseCaptionEvalDataset(ObjaverseCaptionDataset):
    def __getitem__(self, index):
        data = super().__getitem__(index)
        if data != None:
            del data["text_input"]
        return data



class ShapenetCaptionDataset(Object3dCaptionDataset, __DisplMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def get_existing_pc_annotations(self):
        return list(set([f.replace('.npy', '') for f in os.listdir(self.pc_root)]))

    def get_pc_path(self, sample_key):
        return os.path.join(self.pc_root, sample_key+'.npy')
    
    def get_images_path(self, sample_key):
        return [os.path.join(self.vis_root,sample_key, img_path) for img_path in os.listdir(os.path.join(self.vis_root, sample_key))]
        
    def __getitem__(self, index):
        ann = super().__getitem__(index)
        if not isinstance(ann['captions'], list):
            if self.templates:
                ann['objects'] = ann['captions']
                ann['captions'] = [random.choice(self.templates).format(obj) for obj in ann['objects'].split(',')]
            else:
                ann['objects'] = ann['captions']
                ann['captions'] = [random.choice(ann['objects'].split(','))]
        ann['text_input'] = self.text_processor(random.choice(ann['captions']))
        return ann

class ShapenetCaptionInstructDataset(ShapenetCaptionDataset):
    def __getitem__(self, index):
        data = super().__getitem__(index)
        if data != None:
            data['text_output'] = data["text_input"]
            data['text_input'] = self.text_processor("")
        return data

class ShapenetCaptionEvalDataset(ShapenetCaptionDataset):
    def __getitem__(self, index):
        data = super().__getitem__(index)
        if data != None:
            del data["text_input"]
        return data
