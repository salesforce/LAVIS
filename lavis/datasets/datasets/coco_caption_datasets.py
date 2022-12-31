"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import json
import random

from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from lavis.datasets.datasets.caption_datasets import CaptionDataset, CaptionEvalDataset

COCOCapDataset = CaptionDataset


class COCOCapEvalDataset(CaptionEvalDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):
        availabile = False
        while not availabile:
            try:
                ann = self.annotation[index]

                image_path = os.path.join(self.vis_root, ann["image"])
                image = Image.open(image_path).convert("RGB")

                image = self.vis_processor(image)
                img_id = ann["image"].split("/")[-1].strip(".jpg").split("_")[-1]

                availabile = True
            except Exception as e:
                print(f"Error while read file idx {index} in  {e}")
                index = random.randint(0, len(self.annotation) - 1)

        return {
            "image": image,
            "image_id": img_id,
            "instance_id": ann["instance_id"],
        }


class NoCapsEvalDataset(CaptionEvalDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):
        availabile = False
        while not availabile:
            try:
                ann = self.annotation[index]

                image_path = os.path.join(self.vis_root, ann["image"])
                image = Image.open(image_path).convert("RGB")

                image = self.vis_processor(image)
                img_id = ann["img_id"]

                availabile = True
            except Exception as e:
                print(f"Error while read file idx {index} in  {e}")
                index = random.randint(0, len(self.annotation) - 1)

        return {
            "image": image,
            "image_id": img_id,
            "instance_id": ann["instance_id"],
        }
