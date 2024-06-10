"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.common.registry import registry
from lavis.datasets.builders.base_dataset_builder import MultiModalDatasetBuilder
from lavis.datasets.datasets.object3d_captioning_datasets import (
    ObjaverseCaptionDataset,
    ObjaverseCaptionEvalDataset,
    ObjaverseCaptionInstructDataset,
    ShapenetCaptionDataset,
    ShapenetCaptionEvalDataset,
    ShapenetCaptionInstructDataset,
)

@registry.register_builder("objaverse_mm_caption")
class ObjaverseCaptionBuilder(MultiModalDatasetBuilder):
    train_dataset_cls = ObjaverseCaptionDataset
    eval_dataset_cls = ObjaverseCaptionEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/objaverse/defaults_mm_cap.yaml",
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

@registry.register_builder("objaverse_mm_caption_instruct")
class ObjaverseCaptionInstructBuilder(ObjaverseCaptionBuilder):
    train_dataset_cls = ObjaverseCaptionInstructDataset
    eval_dataset_cls = ObjaverseCaptionEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/objaverse/defaults_mm_cap_instruct.yaml",
    }

@registry.register_builder("shapenet_mm_caption")
class ShapenetCaptionBuilder(ObjaverseCaptionBuilder):
    train_dataset_cls = ShapenetCaptionDataset
    eval_dataset_cls = ShapenetCaptionEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/shapenet/defaults_mm_cap.yaml",
    }

@registry.register_builder("shapenet_mm_caption_instruct")
class ShapenetCaptionInstructBuilder(ObjaverseCaptionBuilder):
    train_dataset_cls = ShapenetCaptionInstructDataset
    eval_dataset_cls = ShapenetCaptionEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/shapenet/defaults_mm_cap_instruct.yaml",
    }