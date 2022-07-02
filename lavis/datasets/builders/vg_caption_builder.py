from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.image_text_pair_datasets import ImageTextPairDataset

from lavis.common.registry import registry


@registry.register_builder("vg_caption")
class VGCaptionBuilder(BaseDatasetBuilder):
    train_dataset_cls = ImageTextPairDataset
    type2path = {"default": "configs/datasets/vg/defaults_caption.yaml"}

    vis_urls = {
        "images": {
            "train": [
                "https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip",
                "https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip",
            ]
        }
    }

    def __init__(self, cfg=None):
        super().__init__(cfg)
