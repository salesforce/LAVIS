from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.retrieval_datasets import (
    RetrievalDataset,
    RetrievalEvalDataset,
    VideoRetrievalDataset,
    VideoRetrievalEvalDataset,
)

from lavis.common.registry import registry


@registry.register_builder("msrvtt_retrieval")
class MSRVTTRetrievalBuilder(BaseDatasetBuilder):
    train_dataset_cls = VideoRetrievalDataset
    eval_dataset_cls = VideoRetrievalEvalDataset

    type2path = {"default": "configs/datasets/msrvtt/defaults_ret.yaml"}


@registry.register_builder("didemo_retrieval")
class DiDeMoRetrievalBuilder(BaseDatasetBuilder):
    train_dataset_cls = VideoRetrievalDataset
    eval_dataset_cls = VideoRetrievalEvalDataset

    type2path = {"default": "configs/datasets/didemo/defaults_ret.yaml"}


@registry.register_builder("coco_retrieval")
class COCORetrievalBuilder(BaseDatasetBuilder):
    train_dataset_cls = RetrievalDataset
    eval_dataset_cls = RetrievalEvalDataset

    type2path = {"default": "configs/datasets/coco/defaults_ret.yaml"}


@registry.register_builder("flickr30k")
class Flickr30kBuilder(BaseDatasetBuilder):
    train_dataset_cls = RetrievalDataset
    eval_dataset_cls = RetrievalEvalDataset

    type2path = {"default": "configs/datasets/flickr30k/defaults.yaml"}
