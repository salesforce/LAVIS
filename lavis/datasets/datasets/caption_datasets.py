"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import random
from collections import OrderedDict

from lavis.datasets.datasets.base_dataset import BaseDataset
from PIL import Image


class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": ann["image"],
                "caption": ann["caption"],
                "image": sample["image"],
            }
        )


class CaptionDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann["image_id"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

    def __getitem__(self, index):
        availabile = False
        while not availabile:
            try:
                # TODO this assumes image input, not general enough
                ann = self.annotation[index]

                image_path = os.path.join(self.vis_root, ann["image"])
                image = Image.open(image_path).convert("RGB")

                image = self.vis_processor(image)
                caption = self.text_processor(ann["caption"])

                availabile = True

            except Exception as e:
                print(f"Error while read file idx {index} in  {e}")
                index = random.randint(0, len(self.annotation) - 1)

        return {
            "image": image,
            "text_input": caption,
            "image_id": self.img_ids[ann["image_id"]],
        }


class CaptionEvalDataset(BaseDataset, __DisplMixin):
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
                availabile = True

            except Exception as e:
                print(f"Error while read file idx {index} in  {e}")
                index = random.randint(0, len(self.annotation) - 1)
        return {
            "image": image,
            "image_id": ann["image_id"],
            "instance_id": ann["instance_id"],
        }
