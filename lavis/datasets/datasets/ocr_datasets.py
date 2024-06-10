"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import random
import copy

from PIL import Image
from lavis.datasets.datasets.vqa_datasets import VQADataset


class OCRVQADataset(VQADataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        count_id = 0
        annotations = []
        for ann in self.annotation:
            for q,a in zip(ann['questions'],ann['answers']):
                new_ann = {}
                new_ann = copy.deepcopy(ann)
                new_ann['questions'] = q
                new_ann['answers'] = a   
                new_ann['instance_id'] = count_id
                new_ann['sample_id'] = ann["sample_id"]
                image_id = ann['sample_id'] + '.jpg'
                image_path = os.path.join(self.vis_root, image_id)
                if not os.path.exists(image_path):
                    continue
                count_id+= 1           
                annotations.append(new_ann)
        self.annotation = annotations

    def __getitem__(self, index):
        ann = self.annotation[index]
        image_id = ann['sample_id'] + '.jpg'
        image_path = os.path.join(self.vis_root, image_id)
        try:
            image = Image.open(image_path).convert("RGB")
        except:
            return None
        image = self.vis_processor(image)
        question = self.text_processor(ann["questions"])

        answers = [ann["answers"]]
        # TODO this should be configured better
        weights = [1.]
        
        return {
            "image": image,
            "text_input": question,
            "answers": answers,
            "weights": weights,
            "question_id": ann["sample_id"]
        }

class OCRVQAInstructDataset(OCRVQADataset):
    def __getitem__(self, index):
        data = super().__getitem__(index)
        if data != None:
            data['text_output'] = random.choice(data["answers"])
        return data
    def collater(self, samples):
        data = super().collater(samples)
        data['text_output'] = data['answer']
        return data