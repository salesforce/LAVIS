import os

from PIL import Image
from lavis.datasets.datasets.base_dataset import BaseDataset

from torchvision import datasets


class ImageFolderDataset(BaseDataset):
    def __init__(self, vis_processor, vis_root, classnames=[], **kwargs):
        super().__init__(vis_processor=vis_processor, vis_root=vis_root)

        self.inner_dataset = datasets.ImageFolder(vis_root)

        self.annotation = [
            {"image": elem[0], "label": elem[1], "image_id": elem[0]}
            for elem in self.inner_dataset.imgs
        ]

        self.classnames = classnames

    def __len__(self):
        return len(self.inner_dataset)

    def __getitem__(self, index):
        ann = self.annotation[index]

        img_fn = ann["image"]
        image_path = os.path.join(self.vis_root, img_fn)
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)

        return {
            "image": image,
            "label": ann["label"],
            "image_id": ann["image_id"],
            "instance_id": ann["instance_id"],
        }
