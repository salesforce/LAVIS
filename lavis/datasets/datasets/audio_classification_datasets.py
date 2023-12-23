"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
from collections import OrderedDict
import torch
import copy
import pathlib
import random
import json
import pandas as pd
import torchaudio
import torch
from tqdm import tqdm

from lavis.datasets.datasets.base_dataset import BaseDataset

class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": ann["image"],
                "label": ann["caption"],
                "audio": sample["audio"],
                "audio_path": sample["audio_path"],
                "caption": sample["caption"],
    
            }
        )


class ESC50(BaseDataset, __DisplMixin):
    def __init__(self, **kwargs):
        self.modalities = kwargs['modalities']
        super().__init__(kwargs['vis_processor'], kwargs['text_processor'], kwargs['vis_root'], kwargs['ann_paths'])
        for modality in self.modalities:
            setattr(self, f"{modality}_root", kwargs[f"{modality}_root"])
            setattr(self, f"{modality}_processor", kwargs[f"{modality}_processor"])
            setattr(self, f"existing_{modality}_annotation",getattr(self, f'get_existing_{modality}_annotations')())
        self.classnames = list(set([ann['category'] for ann in self.annotation]))
        self.classnames = [c.replace('_', ' ') for c in self.classnames]
        
    def get_audio_path(self, ann):
        return os.path.join(self.audio_root, ann["sample_id"])
    
    def is_empty_audio(self, ann):
        path = self.get_audio_path(ann)
        try:
            waveform, sr = torchaudio.load(path)

            # Convert to mono if it's stereo
            if waveform.shape[0] == 2:
                waveform = torch.mean(waveform, dim=0)

        except torchaudio.TorchaudioException:
            return True  # Audio loading failed

        return waveform.nelement() == 0
    
    def get_existing_audio_annotations(self):
        return [f for f in os.listdir(self.audio_root)]

    def get_existing_video_annotations(self):
        return os.listdir(self.video_root)
    
    def get_existing_images_annotations(self):
        return os.listdir(self.vis_root)
    
    def get_video_path(self, ann):
        return  pathlib.Path(os.path.join(self.video_root, ann[self.sample_id_key])).resolve()
     
    def get_images_path(self, ann):
        return  pathlib.Path(os.path.join(self.vis_root, ann[self.sample_id_key])).resolve()
    
    def __len__(self):
        return len(self.annotation)
    

    def __getitem__(self, index):
        ann = copy.deepcopy(self.annotation[index])
        ann["sample_id"] = ann["filename"]
        ann['label'] = ann['category'].replace('_', ' ')
        for modality in self.modalities:
            ann[f"{modality}_path"] = getattr(self, f"get_{modality}_path")(ann)
            if isinstance(ann[f"{modality}_path"], list):
                ann[f"{modality}_path"] = random.choice(ann[f"{modality}_path"])
            else:
                ann[modality if 'image' not in modality else 'image'] = getattr(self, f"{'vis' if 'image' in modality else modality}_processor")(ann[f"{modality}_path"])

        if ann["audio"].sum() == 0:
            return None

        return ann

