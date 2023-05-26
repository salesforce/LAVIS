"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import json

from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from lavis.datasets.datasets.caption_datasets import CaptionDataset, CaptionEvalDataset
from lavis.datasets.datasets.base_dataset import BaseDataset

class SherlockDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.vis_root = vis_root

        self.annotation = []
        for ann_path in ann_paths:
            self.annotation.extend(json.load(open(ann_path, "r")))

        self.vis_processor = vis_processor
        self.text_processor = text_processor


    def __getitem__(self, index):

        ann = self.annotation[index]

        input_path_raw = ann["inputs"]["image"]["url"]
        input_path_simple = input_path_raw.split("/")[-2] + "/" + input_path_raw.split("/")[-1]

        image_path = os.path.join(self.vis_root, input_path_simple)
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        caption = self.text_processor(ann["targets"]["inference"])

        return {
            "image": image,
            "text_input": caption,
            "image_id": ann["inputs"]["obs_idx"],   #복잡한거 -> 정수 바꿔서 주므로 간단한거
        }

class SherlockEvalDataset(CaptionEvalDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):
        
        ann = self.annotation[index]

        input_path_raw = ann["inputs"]["image"]["url"]
        input_path_simple = input_path_raw.split("/")[-2] + "/" + input_path_raw.split("/")[-1]

        image_path = os.path.join(self.vis_root, input_path_simple)
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)

        return {
            "image": image,
            "image_id": ann["instance_id"], #복잡
            "instance_id": ann["split_idx"],    #간단한 정수
        }