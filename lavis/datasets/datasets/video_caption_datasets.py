"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
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

        video = self.vis_processor(video_path)
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

    def __getitem__(self, index):

        ann = self.annotation[index]

        vname = ann["video"]
        video_path = os.path.join(self.vis_root, vname)

        video = self.vis_processor(video_path)

        return {
            "video": video,
            "image_id": ann["image_id"],
            "instance_id": ann["instance_id"],
        }
