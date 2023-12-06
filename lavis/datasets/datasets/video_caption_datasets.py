"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import math
from lavis.datasets.datasets.base_dataset import BaseDataset
from lavis.datasets.datasets.caption_datasets import CaptionDataset


class VideoCaptionDataset(CaptionDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):

        ann = self.annotation[index]

        vname = ann["video"]
        video_path = os.path.join(self.vis_root, vname)
        
        try:
            video = self.vis_processor(video_path)
        except:
            print(f"Could not load {video_path}")
            return None
        if video==None:
            return None
        
        caption = self.text_processor(ann["caption"])

        # "image_id" is kept to stay compatible with the COCO evaluation format
        return {
            "video": video,
            "text_input": caption,
            "image_id": self.img_ids[ann["image_id"]],
        }


class VideoCaptionEvalDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        # videos set. do not repeat videos in inference
        ## todo: make it deduplicated because creating annotation file makes 
        seen = set()
        self.annotation = [x for x in self.annotation if x["video"] not in seen and not seen.add(x["image_id"])]
    
    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        ann = self.annotation[index]

        vname = ann["video"]
        video_path = os.path.join(self.vis_root, vname)

        try:
            video = self.vis_processor(video_path)
        except:
            print(f"Could not load {video_path}")
            return None

        return {
            "video": video,
            "image_id": ann["image_id"],
            "instance_id": ann["instance_id"],
        }


class VideoCaptionInstructDataset(VideoCaptionDataset):
    def __getitem__(self, index):
        data = super().__getitem__(index)
        if data != None:
            data['text_output'] = data["text_input"]
            data['text_input'] = self.text_processor("")
        return data



class ClipCaptionDataset(BaseDataset):
    """
    Handles video datasets where subclip of full video needs to be loaded. 
    """
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):

        ann = self.annotation[index]

        vname = ann["video_path"]
        video_path = os.path.join(self.vis_root, vname)
        try:
            video = self.vis_processor(video_path, start_sec=math.floor(ann['ts'][0]), end_sec=math.ceil(ann['ts'][1]))
        except:
            return None


        caption = ann["caption"] if 'caption' in ann else ann["query"]

        image_id = ann['youtube_id'] if 'youtube_id' in ann else ann["video_id"] if "video_id" in ann else vname

        # "image_id" is kept to stay compatible with the COCO evaluation format
        return {
            "video": video,
            "text_input": self.text_processor(caption),
            "image_id": image_id,
            "instance_id": ann['instance_id'],
        }

class ClipCaptionInstructDataset(ClipCaptionDataset):
    def __getitem__(self, index):
        data = super().__getitem__(index)
        if data != None:
            data['text_output'] = data["text_input"]
            data['text_input'] = self.text_processor("")
        return data

class ClipCaptionEvalDataset(ClipCaptionDataset):
    def __getitem__(self, index):
        data = super().__getitem__(index)
        if data != None:
            del data["text_input"]
        return data


class WebVideoCaptionDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
    
    def _get_video(self, index):
        """
        If video does not exist, loop to the next one.
        """
        max_retries = 3
        for _ in range(max_retries):
            ann = self.annotation[index]
            video_path = os.path.join(self.vis_root, f"{ann['videoid']}.mp4")
            try:
                video = self.vis_processor(video_path)
                return video, video_path, ann
            except:
                index = (index + 1) % len(self.annotation)  # Safely loop back to start of annotations
        return None

    def __getitem__(self, index):
        video, video_path, ann = self._get_video(index)
        caption = self.text_processor(ann["name"])

        # "image_id" is kept for compatibility with the COCO evaluation format
        return {
            "video": video,
            "text_input": caption,
            "image_id": ann["videoid"],
            "instance_id": ann["instance_id"],
        }

class WebVideoCaptionInstructDataset(WebVideoCaptionDataset):
    def __getitem__(self, index):
        data = super().__getitem__(index)
        if data != None:
            data['text_output'] = data["text_input"]
            data['text_input'] = self.text_processor("")
        return data
