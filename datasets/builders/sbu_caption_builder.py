from common.registry import registry

from datasets.builders.base_dataset_builder import BaseDatasetBuilder
from datasets.datasets.image_text_pair_datasets import ImageTextPairDataset


@registry.register_builder("sbu_caption")
class SBUCaptionBuilder(BaseDatasetBuilder):
    train_dataset_cls = ImageTextPairDataset

    def __init__(self, cfg):
        super().__init__(cfg)

    @classmethod
    def default_config_path(cls):
        return "configs/datasets/sbu_caption/defaults.yaml"

    def _download_vis(self):
        pass