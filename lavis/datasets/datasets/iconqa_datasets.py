"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from collections import OrderedDict
import json
import os
import torch
import pathlib
import random

from PIL import Image

from lavis.datasets.datasets.vqa_datasets import VQADataset, VQAEvalDataset


class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]
        return OrderedDict(
            {
                "file": ann["image"],
                "question": ann["question"],
                "question_id": ann["question_id"],
                "direct_answers": "; ".join(ann["direct_answers"]),
                "choices": "; ".join(ann["choices"]),
                "correct_choice": ann["choices"][ann["correct_choice_idx"]],
                "image": sample["image"],
            }
        )


class IconQADataset(VQADataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.vis_processor = vis_processor
        self.text_processor = text_processor


    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = pathlib.Path(os.path.join(self.vis_root, ann["image"])).resolve()
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        question = self.text_processor(ann["question"])

        answers = [ann['choices'][ann['answer']]]

        return {
            "image": image,
            "text_input": question,
            "direct_answers": answers,
            "weights": [1],
        }

class IconQAInstructDataset(IconQADataset):
    def __getitem__(self, index):
        data = super().__getitem__(index)
        if data != None:
            data['text_output'] = random.choice(data["direct_answers"])
        return data
        
    def collater(self, samples):
        data = super().collatter(samples)
        data['text_output'] = data['answer']
        return data


class IconQAEvalDataset(VQAEvalDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """

        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.vis_processor = vis_processor
        self.text_processor = text_processor

    def collater(self, samples):
        (
            image_list,
            question_list,
            question_id_list,
            instance_id_list,
            choices_list,
            correct_choice_idx_list,
            direct_answers_list,
        ) = ([], [], [], [], [], [], [])

        for sample in samples:
            image_list.append(sample["image"])
            question_list.append(sample["text_input"])
            question_id_list.append(sample["question_id"])
            instance_id_list.append(sample["instance_id"])
            choices_list.append(sample["choices"])
            correct_choice_idx_list.append(sample["correct_choice_idx"])
            direct_answers_list.append(sample["direct_answers"])

        return {
            "image": torch.stack(image_list, dim=0),
            "text_input": question_list,
            "instance_id": instance_id_list,
            "choices": choices_list,
            "correct_choice_idx": correct_choice_idx_list,
            "direct_answers": direct_answers_list,
        }

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = pathlib.Path(os.path.join(self.vis_root, ann["image"])).resolve()

        answers = [ann['choices'][ann['answer']]]

        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        question = self.text_processor(ann["question"])

        choices = ann["choices"]
        correct_choice_idx = ann["answer"]

        return {
            "image": image,
            "text_input": question,
            "instance_id": ann["instance_id"],
            "choices": choices,
            "correct_choice_idx": correct_choice_idx,
            "direct_answers": answers,
            "question_id": ann["instance_id"]
        }
