from lavis.common.registry import registry

from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.image_text_pair_datasets import ImageTextPairDataset


@registry.register_builder("sbu_caption")
class SBUCaptionBuilder(BaseDatasetBuilder):
    train_dataset_cls = ImageTextPairDataset

    def __init__(self, cfg=None):
        super().__init__(cfg)

    @classmethod
    def default_config_path(cls, type="default"):
        paths = {"default": "lavis/configs/datasets/sbu_caption/defaults.yaml"}

        return paths[type]

    def _download_vis(self):
        pass
