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

class CaptionDatasetInstructWrapper(CaptionDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, instruction_path):
        """
        TODO: Add a probability variable for few-shot examples. E.g.,
            fs_prob = 0.2
            n_fs = k  # the number of examples
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        with open(instruction_path) as f:
            self.instructions = f.read().splitlines()

    def sampleInstruction(self):
        instruction = self.instructions[random.randint(0, len(self.instructions) - 1)]
        return instruction

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        caption = self.text_processor(ann['caption'])

        instruction = self.sampleInstruction()

        return {
            "image": image,
            "text_input": instruction,
            "text_output": caption,
            "image_id": self.img_ids[ann["image_id"]],
        }
