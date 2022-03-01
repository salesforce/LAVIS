from common.registry import registry
from datasets.base_dataset_builder import BaseDatasetBuilder


@registry.register_builder("coco")
class COCOBuilder(BaseDatasetBuilder):
    def __init__(self):
        super().__init__()
    
    @classmethod
    def default_config_path(cls):
        return "configs/datasets/coco/defaults.yaml"