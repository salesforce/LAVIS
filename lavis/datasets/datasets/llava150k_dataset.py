"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.datasets.datasets.base_dataset import BaseDataset
import os
from PIL import Image


class LLaVA150kInstructDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor,ann_paths, vis_root):
        super().__init__(vis_processor=vis_processor, text_processor=text_processor, ann_paths=ann_paths, vis_root=vis_root)
        self.inner_dataset = self.annotation
        self.location = vis_root

    def __len__(self):
        return len(self.inner_dataset)

    def __getitem__(self, index):

        example = self.inner_dataset[index]
        text_input = example['conversations'][0]['value'].replace('<image>', '').strip()
        text_output = example['conversations'][1]['value']
        image_id = example['image']
        image_path = os.path.join(self.location, image_id)
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)
        return {
            "image": image,
            "instance_id":image_id,
            "text_input": self.text_processor(text_input),
            "text_output": self.text_processor(text_output),
            "image_path": image_path
        }
