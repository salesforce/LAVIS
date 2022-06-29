from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder

from lavis.common.registry import registry
from lavis.datasets.datasets.aok_vqa_datasets import AOKVQADataset, AOKVQAEvalDataset
from lavis.datasets.datasets.coco_vqa_datasets import COCOVQADataset, COCOVQAEvalDataset
from lavis.datasets.datasets.vg_vqa_datasets import VGVQADataset


@registry.register_builder("coco_vqa")
class COCOVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = COCOVQADataset
    eval_dataset_cls = COCOVQAEvalDataset

    vis_urls = {
        "images": {
            "train": [
                "http://images.cocodataset.org/zips/train2014.zip",  # md5: 0da8c0bd3d6becc4dcb32757491aca88
                "http://images.cocodataset.org/zips/val2014.zip",  # md5: a3d79f5ed8d289b7a7554ce06a5782b3
            ],
            "test": "http://images.cocodataset.org/zips/test2015.zip",  # md5: 04127eef689ceac55e3a572c2c92f264
        }
    }

    def __init__(self, cfg=None):
        super().__init__(cfg)

    @classmethod
    def default_config_path(cls, type="default"):
        paths = {
            "default": "lavis/configs/datasets/coco/defaults_vqa.yaml",
            "eval": "lavis/configs/datasets/coco/eval_vqa.yaml",
        }

        return paths[type]


@registry.register_builder("vg_vqa")
class VGVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = VGVQADataset

    vis_urls = {
        "images": {
            "train": [
                "https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip",
                "https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip",
            ]
        }
    }

    def __init__(self, cfg=None):
        super().__init__(cfg)

    @classmethod
    def default_config_path(cls, type="default"):
        paths = {"default": "lavis/configs/datasets/vg/defaults_vqa.yaml"}

        return paths[type]


@registry.register_builder("ok_vqa")
class OKVQABuilder(COCOVQABuilder):
    def __init__(self, cfg=None):
        super().__init__(cfg)

    @classmethod
    def default_config_path(cls, type="default"):
        paths = {
            "default": "lavis/configs/datasets/okvqa/defaults.yaml",
        }

        return paths[type]


@registry.register_builder("aok_vqa")
class AOKVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = AOKVQADataset
    eval_dataset_cls = AOKVQAEvalDataset

    def __init__(self, cfg=None):
        super().__init__(cfg)

    @classmethod
    def default_config_path(cls, type="default"):
        paths = {"default": "lavis/configs/datasets/aokvqa/defaults.yaml"}

        return paths[type]

    def _download_vis(self):
        pass
