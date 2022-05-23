from common.registry import registry

from datasets.builders.base_dataset_builder import BaseDatasetBuilder
from datasets.datasets.image_text_pair_datasets import ImageTextPairDataset


@registry.register_builder("conceptual_caption_3m")
class ConceptualCaption3MBuilder(BaseDatasetBuilder):
    train_dataset_cls = ImageTextPairDataset

    def __init__(self, cfg):
        super().__init__(cfg)

    @classmethod
    def default_config_path(cls):
        return "configs/datasets/conceptual_caption/defaults_3m.yaml"

    def _download_vis(self):
        pass


@registry.register_builder("conceptual_caption_12m")
class ConceptualCaption12MBuilder(BaseDatasetBuilder):
    train_dataset_cls = ImageTextPairDataset

    def __init__(self, cfg):
        super().__init__(cfg)

    @classmethod
    def default_config_path(cls):
        return "configs/datasets/conceptual_caption/defaults_12m.yaml"

    def _download_vis(self):
        pass
