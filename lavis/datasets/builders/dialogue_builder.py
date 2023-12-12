"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.common.registry import registry
from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder, MultiModalDatasetBuilder
from lavis.datasets.datasets.avsd_dialogue_datasets import (
    AVSDDialDataset,
    AVSDDialEvalDataset,
    AVSDDialInstructEvalDataset
)
from lavis.datasets.datasets.visdial_dialogue_datasets import (
    VisDialDataset,
    VisDialInstructDataset,
    VisDialEvalDataset,
)

from lavis.datasets.datasets.yt8m_video_dialogue_datasets import YT8MDialDataset
from lavis.datasets.datasets.llava150k_dataset import LLaVA150kInstructDataset


@registry.register_builder("avsd_dialogue")
class AVSDDialBuilder(BaseDatasetBuilder):
    train_dataset_cls = AVSDDialDataset
    eval_dataset_cls = AVSDDialEvalDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/avsd/defaults_dial.yaml"}

@registry.register_builder("visdial")
class VisDialBuilder(BaseDatasetBuilder):
    train_dataset_cls = VisDialDataset
    eval_dataset_cls = VisDialEvalDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/visdial/defaults_dial.yaml"}

@registry.register_builder("visdial_instruct")
class VisDialInstructBuilder(BaseDatasetBuilder):
    train_dataset_cls = VisDialInstructDataset
    eval_dataset_cls = VisDialEvalDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/visdial/defaults_dial_instruct.yaml"}

@registry.register_builder("avsd_mm_dialogue_instruct")
class AVSDDialInstructBuilder(MultiModalDatasetBuilder):
    train_dataset_cls = AVSDDialInstructEvalDataset
    eval_dataset_cls = AVSDDialInstructEvalDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/avsd/defaults_mm_dial_instruct.yaml"}

@registry.register_builder("llava150k_dialogue_instruct")
class LLaVA150kDialInstructBuilder(BaseDatasetBuilder):
    train_dataset_cls = LLaVA150kInstructDataset
    eval_dataset_cls = LLaVA150kInstructDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/llava150k/defaults_dial.yaml"}

@registry.register_builder("yt8m_mm_dialogue")
class YT8MDialBuilder(MultiModalDatasetBuilder):
    train_dataset_cls = YT8MDialDataset
    eval_dataset_cls = YT8MDialDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/yt8m/defaults_mm_dial.yaml"}

