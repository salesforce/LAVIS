"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import random
from lavis.datasets.datasets.base_dataset import BaseDataset
import math

from lavis.datasets.datasets.caption_datasets import CaptionDataset


class VlepVideoDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        existing_videos = [f.replace('.mp4', '') for f in os.listdir(self.vis_root)]
        self.annotation = [ann for ann in self.annotation if ann['vid_name'] in existing_videos]


    def __getitem__(self, index):
        ann = self.annotation[index]

        vname = ann['vid_name']+'.mp4'
        video_path = os.path.join(self.vis_root, vname)

        try:
            video = self.vis_processor(video_path, start_sec=math.floor(ann['ts'][0]), end_sec=math.ceil(ann['ts'][1]))
        except:
            return None
       
        caption = self.text_processor(ann['events'][ann['answer']])

        # "image_id" is kept to stay compatible with the COCO evaluation format
        return {
            "video": video,
            "text_input": self.text_processor(caption),
            "image_id": vname,
            "example_id": ann['example_id'],
            "instance_id": ann["instance_id"]
        }

class VlepVideoInstructDataset(VlepVideoDataset):
    def __getitem__(self, index):
        data = super().__getitem__(index)
        templates = [
            "what is likely to happen next?", 
            "what comes after this?", 
            "where is this leading?",
            "in your estimation, what's the next move?",
            "can you foresee the subsequent events?",
            "based on the video, what might follow?",
            "can you give a glimpse into what might be coming?",
            ]
        if data != None:
            data['text_output'] = data["text_input"]
            data['text_input'] = self.text_processor(random.choice(templates))
        return data

class VlepVideoEvalDataset(VlepVideoDataset):
    def __getitem__(self, index):
        data = super().__getitem__(index)
        if data != None:
            del data["text_input"]
        return data