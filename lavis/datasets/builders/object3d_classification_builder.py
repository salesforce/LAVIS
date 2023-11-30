"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.common.registry import registry
from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder, MultiModalDatasetBuilder
from lavis.datasets.datasets.object3d_classification_datasets import ModelNetClassificationDataset

@registry.register_builder("modelnet40_cls")
class ModelNetClassificationBuilder(MultiModalDatasetBuilder):
    train_dataset_cls = ModelNetClassificationDataset
    eval_dataset_cls = ModelNetClassificationDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/modelnet40/defaults_cls.yaml",
    }