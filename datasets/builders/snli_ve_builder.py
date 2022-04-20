from common.registry import registry

from datasets.builders.base_dataset_builder import BaseDatasetBuilder
from datasets.datasets.snli_ve_datasets import SNLIVisualEntialmentDataset


@registry.register_builder("snli_ve")
class SNLIVisualEntailmentBuilder(BaseDatasetBuilder):
    train_dataset_cls = SNLIVisualEntialmentDataset
    eval_dataset_cls = SNLIVisualEntialmentDataset

    def __init__(self, cfg):
        super().__init__(cfg)

    @classmethod
    def default_config_path(cls):
        return "configs/datasets/snli/defaults.yaml"