from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.vg_vqa_datasets import VGVQADataset

from lavis.common.registry import registry


@registry.register_builder("vg_vqa")
class VGVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = VGVQADataset

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

    @classmethod
    def default_config_path(cls, type="default"):
        paths = {"default": "lavis/configs/datasets/vg/defaults_vqa.yaml"}

        return paths[type]
