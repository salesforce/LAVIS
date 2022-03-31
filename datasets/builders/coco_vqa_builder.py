from datasets.builders.coco_builder import COCOBuilder

from common.registry import registry


@registry.register_builder("coco_vqa")
class COCOVQABuilder(COCOBuilder):
    train_dataset_cls = None
    eval_dataset_cls = None

    def __init__(self, cfg):
        super().__init__(cfg)
    
    @classmethod
    def default_config_path(cls):
        return "configs/datasets/vqa/defaults.yaml"
    
    def build(self):
        import pdb; pdb.set_trace()
        pass