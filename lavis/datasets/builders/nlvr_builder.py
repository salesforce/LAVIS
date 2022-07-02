from lavis.common.registry import registry

from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.nlvr_datasets import NLVRDataset


@registry.register_builder("nlvr")
class NLVRBuilder(BaseDatasetBuilder):
    train_dataset_cls = NLVRDataset
    eval_dataset_cls = NLVRDataset

    type2path = {"default": "configs/datasets/nlvr/defaults.yaml"}

    def __init__(self, cfg=None):
        super().__init__(cfg)

    def _download_vis(self):
        pass
