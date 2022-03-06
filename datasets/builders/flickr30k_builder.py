from common.registry import registry
from datasets.builders.retrieval_builder import RetrievalBuilder


@registry.register_builder("flickr30k")
class Flickr30kBuilder(RetrievalBuilder):
    def __init__(self, cfg):
        super().__init__(cfg)
    
    @classmethod
    def default_config_path(cls):
        return "configs/datasets/flickr30k/defaults.yaml"

    def _download_vis(self):
        pass