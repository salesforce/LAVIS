from common.registry import registry
from datasets.builders.base_dataset_builder import BaseDatasetBuilder


@registry.register_builder("flickr30k")
class Flickr30kBuilder(BaseDatasetBuilder):
    def __init__(self):
        super().__init__()
    
    @classmethod
    def default_config_path(cls):
        return "configs/datasets/flickr30k/defaults.yaml"