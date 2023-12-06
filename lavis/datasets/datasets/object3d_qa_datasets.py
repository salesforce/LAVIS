"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import copy
import random
from PIL import Image
import torch

from lavis.datasets.datasets.object3d_captioning_datasets import Object3dCaptionDataset

class ObjaverseQADataset(Object3dCaptionDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_binary = kwargs.get('add_binary', False)
        self.binary_templates = ["do you see {}?", "is this {}?", "does the 3d model contain {}?"]
        self.remove_model_answer = kwargs.get('remove_model_answer', False)
        if self.remove_model_answer:
            self.annotation = [ann for ann in self.annotation if 'model' not in ann['answer']]
    
    def get_existing_pc_annotations(self):
        return list(set(os.listdir(self.pc_root)).intersection(set(ann['sample_id'] for ann in self.annotation)))

    def get_pc_path(self, sample_key):
        return os.path.join(self.pc_root, sample_key, '{}_{}.npz'.format(sample_key, self.npoints))
       
    def get_images_path(self, sample_key):
        # data downloaded from: https://huggingface.co/datasets/tiange/Cap3D/tree/main/RenderedImage_zips
        return [os.path.join(self.vis_root, f'compressed_imgs_view{i}/Cap3D_imgs_view{i}/', sample_key+f'_{i}.jpeg') for i in range(8)]
        
    def __getitem__(self, index):
        ann = copy.deepcopy(self.annotation[index])
        for modality in self.modalities:
            ann[f"{modality}_path"] = getattr(self, f"get_{modality}_path")(ann['sample_id'])
            if type(ann[f"{modality}_path"]) == list:
                ann[f"{modality}_path"] = random.choice(ann[f"{modality}_path"])
            if 'image' in modality:
                ann['image'] = self.vis_processor(Image.open(ann[f"image_path"]))
            else:
                ann[modality] = getattr(self, f"{modality}_processor")(ann[f"{modality}_path"]).to(torch.float32)
        
        if self.add_binary and random.randint(0,10) < 3:
            yes_answer = random.randint(0,10)<5
            if not yes_answer:
                caption_index = random.choice(list(set(range(len(self.annotation))).difference(set([index]))))
                caption = self.annotation[caption_index]['caption']
            else:
                caption = ann['caption']
            
            question = random.choice(self.binary_templates).format(caption)
            answer = 'yes' if yes_answer else 'no'
            ann['text_input'] = self.text_processor(question)
            ann['text_output'] = answer

        else:
            ann['text_input'] = self.text_processor(ann['question'])
            ann['text_output'] =  ann['answer']

        ann['answers'] =  [ann['text_output']]
        ann['question_id'] = ann['instance_id']
        return ann