"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import random

from collections import OrderedDict

from lavis.datasets.datasets.multimodal_classification_datasets import (
    MultimodalClassificationDataset,
)
from PIL import Image


class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file_L": ann["images"][0],
                "file_R": ann["images"][1],
                "sentence": ann["sentence"],
                "label": ann["label"],
                "image": [sample["image0"], sample["image1"]],
            }
        )


class NLVRDataset(MultimodalClassificationDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.class_labels = self._build_class_labels()

    def _build_class_labels(self):
        return {"False": 0, "True": 1}

    @staticmethod
    def _flip(samples):
        sentence = samples["text_input"]
        image0, image1 = samples["image0"], samples["image1"]

        if "left" not in sentence and "right" not in sentence:
            if random.random() < 0.5:
                image0, image1 = image1, image0
        else:
            if random.random() < 0.5:
                sentence = sentence.replace("left", "[TEMP_TOKEN]")
                sentence = sentence.replace("right", "left")
                sentence = sentence.replace("[TEMP_TOKEN]", "right")

                image0, image1 = image1, image0

        samples["text_input"] = sentence
        samples["image0"] = image0
        samples["image1"] = image1

        return samples

    def __getitem__(self, index):
        ann = self.annotation[index]

        image0_path = os.path.join(self.vis_root, ann["images"][0])
        image0 = Image.open(image0_path).convert("RGB")
        image0 = self.vis_processor(image0)

        image1_path = os.path.join(self.vis_root, ann["images"][1])
        image1 = Image.open(image1_path).convert("RGB")
        image1 = self.vis_processor(image1)

        sentence = self.text_processor(ann["sentence"])
        label = self.class_labels[ann["label"]]

        return self._flip(
            {
                "image0": image0,
                "image1": image1,
                "text_input": sentence,
                "label": label,
                # "image_id": ann["image_id"],
                "instance_id": ann["instance_id"],
            }
        )


class NLVREvalDataset(NLVRDataset):
    @staticmethod
    def _flip(samples):
        return samples
