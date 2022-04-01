from datasets.builders.coco_builder import COCOBuilder

from common.registry import registry
from datasets.datasets.vqa_datasets import VQADataset, VQAEvalDataset


@registry.register_builder("coco_vqa")
class COCOVQABuilder(COCOBuilder):
    train_dataset_cls = VQADataset
    eval_dataset_cls = VQAEvalDataset

    def __init__(self, cfg):
        super().__init__(cfg)
    
    @classmethod
    def default_config_path(cls):
        return "configs/datasets/coco/defaults_vqa.yaml"