from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.avsd_dialogue_datasets import (
    AVSDDialDataset, #COCOCapDataset,
    AVSDDialEvalDataset #COCOCapEvalDataset,
    #NoCapsEvalDataset,
)

from lavis.common.registry import registry

#from lavis.datasets.datasets.video_caption_datasets import (
#    VideoCaptionDataset,
#    VideoCaptionEvalDataset,
#)


@registry.register_builder("avsd_dialogue")
class AVSDDialBuilder(BaseDatasetBuilder):
    train_dataset_cls = AVSDDialDataset #COCOCapDataset
    eval_dataset_cls = AVSDDialEvalDataset #COCOCapEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/avsd/defaults_dial.yaml" #coco/defaults_cap.yaml",
    }