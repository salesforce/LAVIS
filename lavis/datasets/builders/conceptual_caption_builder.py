from lavis.common.registry import registry

from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.image_text_pair_datasets import ImageTextPairDataset


@registry.register_builder("conceptual_caption_3m")
class ConceptualCaption3MBuilder(BaseDatasetBuilder):
    train_dataset_cls = ImageTextPairDataset

    def __init__(self, cfg=None):
        super().__init__(cfg)

    @classmethod
    def default_config_path(cls, type="default"):
        paths = {
            "default": "lavis/configs/datasets/conceptual_caption/defaults_3m.yaml"
        }

        return paths[type]

    def _download_vis(self):
        pass


@registry.register_builder("conceptual_caption_12m")
class ConceptualCaption12MBuilder(BaseDatasetBuilder):
    train_dataset_cls = ImageTextPairDataset

    def __init__(self, cfg=None):
        super().__init__(cfg)

    @classmethod
    def default_config_path(cls, type="default"):
        paths = {
            "default": "lavis/configs/datasets/conceptual_caption/defaults_12m.yaml"
        }

        return paths[type]

    def _download_vis(self):
        pass
