import os

from lavis.datasets.datasets.base_dataset import BaseDataset
from PIL import Image


class ImageTextPairDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, image_root, ann_paths):
        """
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, image_root, ann_paths)

    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
        ann = self.annotation[index]

        image_path = os.path.join(self.image_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        caption = self.text_processor(ann["caption"])

        return {"image": image, "text_input": caption}
