"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
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
                "file": os.path.basename(ann["image"]),
                "sentence": ann["sentence"],
                "label": ann["label"],
                "image": sample["image"],
            }
        )


class SNLIVisualEntialmentDataset(MultimodalClassificationDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        self.class_labels = self._build_class_labels()
        self.classnames = list(self.class_labels.keys())

    def _build_class_labels(self):
        return {"contradiction": 0, "neutral": 1, "entailment": 2}

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_id = ann["image"]
        image_path = os.path.join(self.vis_root, "%s.jpg" % image_id)
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        sentence = self.text_processor(ann["sentence"])

        return {
            "image": image,
            "text_input": sentence,
            "label": self.class_labels[ann["label"]],
            "image_id": image_id,
            "instance_id": ann["instance_id"],
        }

class SNLIVisualEntialmentInstructDataset(SNLIVisualEntialmentDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        self.classnames = ['no', 'maybe', 'yes']

    def __getitem__(self, index):
        data = super().__getitem__(index)
        if data != None:
            data["prompt"] = self.text_processor("based on the given the image is {} true?")
            data["answer"] = self.classnames[data["label"]]
            data["label"] = self.classnames[data["label"]]
            data["question_id"] = data["instance_id"]
        return data
