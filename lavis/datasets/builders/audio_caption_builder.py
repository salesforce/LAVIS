"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.common.registry import registry
from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder, MultiModalDatasetBuilder

from lavis.datasets.datasets.audio_captioning_datasets import (
    AudioSetDataset,
    AudioSetEvalDataset,
    AudioSetInstructDataset,
    AudioCapsDataset,
    AudioCapsEvalDataset,
    AudioCapsInstructDataset,
    ClothoV2Dataset,
    ClothoV2InstructDataset,
    ClothoV2EvalDataset,
    AudioLanguagePretrainDataset,
    AudioLanguagePretrainEvalDataset,
    AudioLanguagePretrainInstructDataset
)


class AudioCapBuilder(MultiModalDatasetBuilder):
    train_dataset_cls = AudioSetDataset
    eval_dataset_cls = AudioSetEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/audioset/defaults_mm_cap.yaml",
    }


    def build(self):
        datasets = super().build()
        build_info = self.config.build_info
        for split,ds in datasets.items():
            # TODO: add option to download templates
            templates = build_info.get('templates')
            if templates == None:
                ds._build_templates(None)
            else:
                ds._build_templates(build_info.templates.storage)
        return datasets

@registry.register_builder("audioset_mm_caption")
class AudioSetBuilder(AudioCapBuilder):
    train_dataset_cls = AudioSetDataset
    eval_dataset_cls = AudioSetEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/audioset/defaults_mm_cap.yaml",
    }

@registry.register_builder("audioset_mm_caption_instruct")
class AudioSetInstructBuilder(AudioCapBuilder):
    train_dataset_cls = AudioSetInstructDataset
    eval_dataset_cls = AudioSetEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/audioset/defaults_mm_cap_instruct.yaml",
    }

@registry.register_builder("audiocaps_mm_caption")
class AudioCapsCapBuilder(AudioCapBuilder):
    train_dataset_cls = AudioCapsDataset
    eval_dataset_cls = AudioCapsEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/audiocaps/defaults_mm_cap.yaml",
    }

@registry.register_builder("audiocaps_mm_caption_instruct")
class AudioCapsInstructCapBuilder(AudioCapBuilder):
    train_dataset_cls = AudioCapsInstructDataset
    eval_dataset_cls = AudioCapsEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/audiocaps/defaults_mm_cap_instruct.yaml",
    }

@registry.register_builder("clothov2")
class ClothoCapInstructBuilder(MultiModalDatasetBuilder):
    train_dataset_cls = ClothoV2Dataset
    eval_dataset_cls = ClothoV2EvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/clotho/defaults_mm_cap.yaml",
    }

@registry.register_builder("clothov2_instruct")
class ClothoCapInstructBuilder(MultiModalDatasetBuilder):
    train_dataset_cls = ClothoV2InstructDataset
    eval_dataset_cls = ClothoV2EvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/clotho/defaults_mm_cap_instruct.yaml",
    }


@registry.register_builder("wavcaps_mm_caption")
class WavCapsCapBuilder(AudioCapBuilder):
    train_dataset_cls = AudioLanguagePretrainDataset
    eval_dataset_cls = AudioLanguagePretrainEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/wavcaps/defaults_mm_cap.yaml",
    }



@registry.register_builder("wavcaps_mm_caption_instruct")
class WavCapsCapInstructBuilder(AudioCapBuilder):
    train_dataset_cls = AudioLanguagePretrainInstructDataset
    eval_dataset_cls = AudioLanguagePretrainEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/wavcaps/defaults_mm_cap_instruct.yaml",
    }


