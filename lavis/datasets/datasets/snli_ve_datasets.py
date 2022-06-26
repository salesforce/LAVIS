import os

from PIL import Image
from lavis.datasets.datasets.multimodal_classification_datasets import (
    MultimodalClassificationDataset,
)


class SNLIVisualEntialmentDataset(MultimodalClassificationDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.class_labels = self._build_class_labels()

        self.add_unique_ids(key="instance_id")

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
