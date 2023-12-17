"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""


import torch
import copy
import os
import random
import json
from PIL import Image
from lavis.datasets.datasets.base_dataset import BaseDataset

class VALORCaptionDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(kwargs['vis_processor'], kwargs['text_processor'], kwargs['vis_root'], kwargs['ann_paths'])

        self.modalities = kwargs['modalities']

        for modality in self.modalities:
            if 'image' in modality:
                setattr(self, f"existing_{modality}_annotation",getattr(self, f'get_existing_{modality}_annotations')())
                continue
            setattr(self, f"{modality}_root", kwargs[f"{modality}_root"])
            setattr(self, f"{modality}_processor", kwargs[f"{modality}_processor"])
            setattr(self, f"existing_{modality}_annotation",getattr(self, f'get_existing_{modality}_annotations')())

        self.sample_ids = set.intersection(*[set(getattr(self, f"existing_{modality}_annotation")) for modality in self.modalities])
        self.annotation = [ann for ann in self.annotation if ann['video_id'].replace('000', '0') in self.sample_ids]
        seen = set()
        self.annotation = [x for x in self.annotation if x["video_id"] not in seen and not seen.add(x["video_id"])]
    
    def __len__(self):
        return len(self.annotation)
    
    def get_existing_audio_annotations(self):
        return ['.'.join(f.split('.')[:-1]) for f in os.listdir(self.audio_root)]
    
    def get_existing_video_annotations(self):
        return ['.'.join(f.split('.')[:-1]) for f in os.listdir(self.video_root)]
    
    
    def get_audio_path(self, ann):
        return os.path.join(self.audio_root, f'{ann["video_id"].replace("000", "0")}.mp4')
    
    def get_video_path(self, ann):
        return os.path.join(self.video_root, f'{ann["video_id"].replace("000", "0")}.mp4')


    def __getitem__(self, index):
        ann = copy.deepcopy(self.annotation[index])
        ann["sample_id"] = ann["video_id"]
        ann["text_input"] =  self.text_processor(ann['desc'])
        for modality in self.modalities:
            ann[f"{modality}_path"] = getattr(self, f"get_{modality}_path")(ann)
            if type(ann[f"{modality}_path"]) == list:
                ann[f"{modality}_path"] = random.choice(ann[f"{modality}_path"])
            if 'image' in modality:
                ann['image'] = self.vis_processor(Image.open(ann[f"images_path"]))
            else:
                ann[modality] = getattr(self, f"{modality}_processor")(ann[f"{modality}_path"]).to(torch.float32)

        ann["caption"] =  ann["text_input"] 
        ann["image_id"] = ann["video_id"]

        
        return ann


class VALORCaptionEvalDataset(VALORCaptionDataset):
    def __getitem__(self, index):
        data = super().__getitem__(index)
        if data != None:
            del data['text_input']
            del data['caption']
        return data


class VALORCaptionInstructDataset(VALORCaptionDataset):
    def __getitem__(self, index):
        data = super().__getitem__(index)
        if data != None:
            data['text_output'] = data["text_input"]
            data['text_input'] = self.text_processor("")
        return data
