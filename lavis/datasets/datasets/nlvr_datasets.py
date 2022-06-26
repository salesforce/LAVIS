import os

from PIL import Image
from lavis.datasets.datasets.multimodal_classification_datasets import (
    MultimodalClassificationDataset,
)


class NLVRDataset(MultimodalClassificationDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.class_labels = self._build_class_labels()

        self.add_unique_ids()

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
            "image_id": ann["image_id"],
        }
