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

from torch.utils.data import Dataset
from lavis.datasets.datasets.caption_datasets import CaptionDataset, CaptionEvalDataset


class CaptionDatasetInstructWrapper(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, instruction_path):
        """
        TODO:
            Add a probability variable for few-shot examples. E.g.,
            fs_prob = 0.2
            n_fs = k  # the number of examples
        """
        super().__init__()

        with open(instruction_path) as f:
            self.instructions = f.read().splitlines()

        self.dataset = CaptionDataset(vis_processor, text_processor, vis_root, ann_paths)

    def sampleInstruction(self):
        instruction = self.instructions[random.randint(0, len(self.instructions) - 1)]
        return instruction

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset.__getitem__(index)

        instruction = self.sampleInstruction()

        return {
            "image": data["image"],
            "text_input": instruction,
            "text_output": data["text_input"],
            "image_id": data["image_id"],
        }