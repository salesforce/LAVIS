"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.common.registry import registry
from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder, MultiModalDatasetBuilder
from lavis.datasets.datasets.discriminatory_reasoning_datasets import DisCRnDataset



@registry.register_builder("image_pc_discrn")
class DiscrnImagePcBuilder(MultiModalDatasetBuilder):
    eval_dataset_cls = DisCRnDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/discriminatory_reasoning/defaults_mm_image_pc.yaml",
    }

@registry.register_builder("audio_video_discrn")
class DiscrnAudioVideoBuilder(MultiModalDatasetBuilder):
    eval_dataset_cls = DisCRnDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/discriminatory_reasoning/defaults_mm_audio_video.yaml",
    }
