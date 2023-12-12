"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.common.registry import registry
from lavis.common.utils import get_cache_path
from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder, MultiModalDatasetBuilder
from lavis.datasets.datasets.video_vqa_datasets import VideoQADataset, VideoQAInstructDataset
from lavis.datasets.datasets.music_avqa import MusicAVQAInstructDataset, MusicAVQADataset


class VideoQABuilder(BaseDatasetBuilder):
    train_dataset_cls = VideoQADataset
    eval_dataset_cls = VideoQADataset

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
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/msrvtt/defaults_qa.yaml",
    }


@registry.register_builder("msvd_qa")
class MSVDQABuilder(VideoQABuilder):
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/msvd/defaults_qa.yaml",
    }


@registry.register_builder("msrvtt_qa_instruct")
class MSRVTTQAInstructBuilder(VideoQABuilder):
    train_dataset_cls = VideoQAInstructDataset
    eval_dataset_cls = VideoQAInstructDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/msrvtt/defaults_qa_instruct.yaml",
    }


@registry.register_builder("msvd_qa_instruct")
class MSVDQAInstructBuilder(VideoQABuilder):
    train_dataset_cls = VideoQAInstructDataset
    eval_dataset_cls = VideoQAInstructDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/msvd/defaults_qa_instruct.yaml",
    }

@registry.register_builder("musicavqa_mm")
class MusicAVQABuilder(MultiModalDatasetBuilder):
    train_dataset_cls = MusicAVQADataset
    eval_dataset_cls = MusicAVQADataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/music_avqa/defaults_mm_qa.yaml"}

@registry.register_builder("musicavqa_mm_instruct")
class MusicAVQAInstructBuilder(MultiModalDatasetBuilder):
    train_dataset_cls = MusicAVQAInstructDataset
    eval_dataset_cls = MusicAVQAInstructDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/music_avqa/defaults_mm_qa_instruct.yaml"}