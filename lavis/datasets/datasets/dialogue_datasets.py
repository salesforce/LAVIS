"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
from collections import OrderedDict

from PIL import Image

from lavis.datasets.datasets.base_dataset import BaseDataset

import json
import copy


class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": ann["image"],
                "dialogue": ann["dialogue"],
                "image": sample["image"],
            }
        )


class DialogueDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """

        self.vis_root = vis_root

        self.annotation = []
        for ann_path in ann_paths:
            dialogs = json.load(open(ann_path, "r"))["dialogs"]
            for dialog in dialogs:
                all_turns = dialog["dialog"]
                dialogue_context = []
                for turn in all_turns:
                    dialog_instance = copy.deepcopy(dialog)
                    question = turn["question"]
                    answer = turn["answer"]

                    dialog_instance["dialog"] = copy.deepcopy(dialogue_context)
                    dialog_instance["question"] = question
                    dialog_instance["answer"] = answer
                    self.annotation.append(dialog_instance)
                    dialogue_context.append(turn)

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self._add_instance_ids()

        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann["image_id"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

    def __getitem__(self, index):

        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        caption = self.text_processor(ann["caption"])

        return {
            "image": image,
            "text_input": caption,
            "image_id": self.img_ids[ann["image_id"]],
        }


class DialogueEvalDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """

        self.vis_root = vis_root

        self.annotation = []
        for ann_path in ann_paths:
            dialogs = json.load(open(ann_path, "r"))["dialogs"]
            for dialog in dialogs:
                all_turns = dialog["dialog"]
                dialogue_context = all_turns[:-1]
                last_turn = all_turns[-1]

                question = last_turn["question"]
                answer = last_turn["answer"]

                dialog["dialog"] = dialogue_context
                dialog["question"] = question
                dialog["answer"] = answer

                self.annotation.append(dialog)

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self._add_instance_ids()

        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann["image_id"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

    def __getitem__(self, index):

        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)

        return {
            "image": image,
            "image_id": ann["image_id"],
            "instance_id": ann["instance_id"],
        }
