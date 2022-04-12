from datasets.datasets.coco_caption_datasets import COCOCapDataset, COCOCapEvalDataset
from datasets.builders.coco_builder import COCOBuilder

from common.registry import registry


@registry.register_builder("coco_caption")
class COCOCapBuilder(COCOBuilder):
    train_dataset_cls = COCOCapDataset
    eval_dataset_cls = COCOCapEvalDataset

    def __init__(self, cfg):
        super().__init__(cfg)

    @classmethod
    def default_config_path(cls):
        return "configs/datasets/coco/defaults_cap.yaml"
