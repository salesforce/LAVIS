import os

from PIL import Image
from datasets.datasets.multimodal_classification_datasets import MultimodalClassificationDataset


class SNLIVisualEntialmentDataset(MultimodalClassificationDataset):
    def __init__(self, vis_processor, text_processor, image_root, ann_paths):
        super().__init__(vis_processor, text_processor, image_root, ann_paths)

        self.class_labels = {
            'contradiction':0,
            'neutral':1,
            'entailment':2
        }

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_id = ann['image']
        image_path = os.path.join(self.image_root, "%s.jpg" % image_id)
        image = Image.open(image_path).convert("RGB")   

        image = self.vis_processor(image)          
        sentence = self.text_processor(ann["sentence"])

        return {
            "image": image,
            "text_input": sentence,
            "label": self.class_labels[ann["label"]],
            "image_id": image_id
        }