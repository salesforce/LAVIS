from common.registry import registry
from datasets.builders.retrieval_builder import RetrievalBuilder


@registry.register_builder("coco_retrieval")
class COCORetrievalBuilder(RetrievalBuilder):
    def __init__(self, cfg):
        super().__init__(cfg)
    
    @classmethod
    def default_config_path(cls):
        return "configs/datasets/coco/defaults.yaml"
    
    def _download_vis(self):
        pass