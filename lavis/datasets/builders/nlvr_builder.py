from common.registry import registry

from datasets.builders.base_dataset_builder import BaseDatasetBuilder
from datasets.datasets.nlvr_datasets import NLVRDataset


@registry.register_builder("nlvr")
class NLVRBuilder(BaseDatasetBuilder):
    train_dataset_cls = NLVRDataset
    eval_dataset_cls = NLVRDataset

    def __init__(self, cfg):
        super().__init__(cfg)

    @classmethod
    def default_config_path(cls):
        return "configs/datasets/nlvr/defaults.yaml"

    def _download_vis(self):
        pass
