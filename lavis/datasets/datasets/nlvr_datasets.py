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

        return {
            "image0": image0,
            "image1": image1,
            "text_input": sentence,
            "label": label,
            # "image_id": ann["image_id"],
            "instance_id": ann["instance_id"],
        }
