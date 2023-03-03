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
from lavis.datasets.datasets.caption_datasets import CaptionDataset


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

        self.dataset = None

    def sample_instruction(self):
        instruction = self.instructions[random.randint(0, len(self.instructions) - 1)]
        return instruction

    def process_instruction(self, instruction):
        instruction = instruction.lower()
        instruction = instruction.rstrip("\n")
        instruction = instruction.strip(" ")
        return instruction

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset.__getitem__(index)
        instruction = self.sample_instruction()
        instruction = self.process_instruction(instruction)

        return {
            "image": data["image"],
            "text_input": instruction,
            "text_output": data["text_input"],
        }
