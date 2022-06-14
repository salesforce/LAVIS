import json
from lavis.common.utils import get_cache_path

from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder

from lavis.common.registry import registry
from lavis.datasets.datasets.video_vqa_datasets import VideoQADataset


class VideoQABuilder(BaseDatasetBuilder):
    train_dataset_cls = VideoQADataset
    eval_dataset_cls = VideoQADataset

    def __init__(self, cfg=None):
        super().__init__(cfg)

    def build(self):
        datasets = super().build()

        ans2label = self.config.build_info.annotations.get("ans2label")
        if ans2label is None:
            raise ValueError("ans2label is not specified in build_info.")

        ans2label = get_cache_path(ans2label.storage)

        for split in datasets:
            datasets[split]._build_class_labels(ans2label)

        return datasets


@registry.register_builder("msrvtt_qa")
class MSRVTTQABuilder(VideoQABuilder):
    @classmethod
    def default_config_path(cls, type="default"):
        paths = {
            "default": "lavis/configs/datasets/msrvtt/defaults_qa.yaml",
        }

        return paths[type]

    def _download_vis(self):
        pass


@registry.register_builder("msvd_qa")
class MSVDQABuilder(VideoQABuilder):
    @classmethod
    def default_config_path(cls, type="default"):
        paths = {
            "default": "lavis/configs/datasets/msvd/defaults_qa.yaml",
        }

        return paths[type]

    def _download_vis(self):
        pass
