"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.common.registry import registry
from lavis.datasets.builders.audio_caption_builder import AudioCapBuilder
from lavis.datasets.datasets.audio_qa_datasets import AudioCapsQADataset, ClothoQADataset

@registry.register_builder("audiocaps_mm_qa")
class AudioCapsQABuilder(AudioCapBuilder):
    train_dataset_cls = AudioCapsQADataset
    eval_dataset_cls = AudioCapsQADataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/audiocaps/defaults_mm_qa.yaml",
    }

@registry.register_builder("clotho_qa")
class ClothoQABuilder(AudioCapBuilder):
    train_dataset_cls = ClothoQADataset
    eval_dataset_cls = ClothoQADataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/clotho/defaults_mm_qa.yaml",
    }