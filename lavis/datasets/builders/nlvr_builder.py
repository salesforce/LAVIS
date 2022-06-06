from lavis.common.registry import registry

from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.nlvr_datasets import NLVRDataset


@registry.register_builder("nlvr")
class NLVRBuilder(BaseDatasetBuilder):
    train_dataset_cls = NLVRDataset
    eval_dataset_cls = NLVRDataset

    def __init__(self, cfg=None):
        super().__init__(cfg)

    @classmethod
    def default_config_path(cls, type="default"):
        paths = {"default": "lavis/configs/datasets/nlvr/defaults.yaml"}

        return paths[type]

    def _download_vis(self):
        pass
