"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import os
from collections import Counter
from PIL import Image
from lavis.datasets.datasets.vqa_datasets import VQAEvalDataset

class VizWizEvalDataset(VQAEvalDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):
        ann = self.annotation[index]
        if 'val' in ann["image"]:
            image_path = os.path.join(self.vis_root.replace('images', 'val'), ann["image"])
        else:
            image_path = os.path.join(self.vis_root.replace('images', 'test'), ann["image"])
            
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        question = self.text_processor(ann["question"])

        if "answers" in ann:
            num_annotators = len(ann["answers"])
            answers = [item['answer'] for item in ann["answers"]]
            answer_counts = Counter(answers)
            answers = list(set(answers))
            weights = [answer_counts[ans]/num_annotators for ans in answers]
        else:
            # test
            return {
            "image": image,
            "question_id": ann["image"],
            "instance_id": ann["instance_id"],
            "text_input": question,
            }

        return {
            "image": image,
            "text_input": question,
            "instance_id": ann["instance_id"],
            "question_id": ann["instance_id"],
            "weights": weights,
            "answer": answers
        }
