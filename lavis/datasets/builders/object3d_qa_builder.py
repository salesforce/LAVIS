"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.common.registry import registry
from lavis.datasets.builders.object3d_caption_builder import ObjaverseCaptionBuilder
from lavis.datasets.datasets.object3d_qa_datasets import ObjaverseQADataset

@registry.register_builder("objaverse_mm_qa")
class ObjaverseQABuilder(ObjaverseCaptionBuilder):
    train_dataset_cls = ObjaverseQADataset
    eval_dataset_cls = ObjaverseQADataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/objaverse/defaults_mm_qa.yaml",
    }