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

    def __init__(self, cfg=None):
        super().__init__(cfg)

    def _download_vis(self):
        pass


@registry.register_builder("didemo_retrieval")
class DiDeMoRetrievalBuilder(BaseDatasetBuilder):
    train_dataset_cls = VideoRetrievalDataset
    eval_dataset_cls = VideoRetrievalEvalDataset

    type2path = {"default": "configs/datasets/didemo/defaults_ret.yaml"}

    def __init__(self, cfg=None):
        super().__init__(cfg)

    def _download_vis(self):
        pass


@registry.register_builder("coco_retrieval")
class COCORetrievalBuilder(BaseDatasetBuilder):
    train_dataset_cls = RetrievalDataset
    eval_dataset_cls = RetrievalEvalDataset

    type2path = {"default": "configs/datasets/coco/defaults_ret.yaml"}

    vis_urls = {
        "images": {
            "train": "http://images.cocodataset.org/zips/train2014.zip",  # md5: 0da8c0bd3d6becc4dcb32757491aca88
            "val": "http://images.cocodataset.org/zips/val2014.zip",  # md5: a3d79f5ed8d289b7a7554ce06a5782b3
            "test": "http://images.cocodataset.org/zips/test2014.zip",  # md5: 04127eef689ceac55e3a572c2c92f264
        }
    }

    def __init__(self, cfg=None):
        super().__init__(cfg)


@registry.register_builder("flickr30k")
class Flickr30kBuilder(BaseDatasetBuilder):
    train_dataset_cls = RetrievalDataset
    eval_dataset_cls = RetrievalEvalDataset

    type2path = {"default": "configs/datasets/flickr30k/defaults.yaml"}

    def __init__(self, cfg=None):
        super().__init__(cfg)

    def _download_vis(self):
        pass
