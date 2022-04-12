from datasets.builders.coco_builder import COCOBuilder

from common.registry import registry
from datasets.datasets.coco_vqa_datasets import COCOVQADataset, COCOVQAEvalDataset


@registry.register_builder("coco_vqa")
class COCOVQABuilder(COCOBuilder):
    train_dataset_cls = COCOVQADataset
    eval_dataset_cls = COCOVQAEvalDataset

    def __init__(self, cfg):
        super().__init__(cfg)

    @classmethod
    def default_config_path(cls):
        return "configs/datasets/coco/defaults_vqa.yaml"
