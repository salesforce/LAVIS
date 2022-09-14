from lavis.common.registry import registry
from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.avsd_dialogue_datasets import (
    AVSDDialDataset,
    AVSDDialEvalDataset,
)


@registry.register_builder("avsd_dialogue")
class AVSDDialBuilder(BaseDatasetBuilder):
    train_dataset_cls = AVSDDialDataset
    eval_dataset_cls = AVSDDialEvalDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/avsd/defaults_dial.yaml"}
