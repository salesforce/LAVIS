"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.common.registry import registry
from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.builders.base_instruct_dataset_builder import BaseInstructDatasetBuilder
from lavis.datasets.datasets.nlvr_datasets import NLVRDataset, NLVREvalDataset
from lavis.datasets.datasets.snli_ve_datasets import SNLIVisualEntialmentDataset
from lavis.datasets.datasets.instruct_wrappers.snli_ve_instruct_datasets import SNLIVisualEntialmentInstructDataset

@registry.register_builder("nlvr")
class NLVRBuilder(BaseDatasetBuilder):
    train_dataset_cls = NLVRDataset
    eval_dataset_cls = NLVREvalDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/nlvr/defaults.yaml"}


@registry.register_builder("snli_ve")
class SNLIVisualEntailmentBuilder(BaseDatasetBuilder):
    train_dataset_cls = SNLIVisualEntialmentDataset
    eval_dataset_cls = SNLIVisualEntialmentDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/snli_ve/defaults.yaml"}

@registry.register_builder("snli_ve_instruct")
class SNLIVisualEntailmentInstructBuilder(BaseInstructDatasetBuilder):
    train_dataset_cls = SNLIVisualEntialmentInstructDataset
    eval_dataset_cls = SNLIVisualEntialmentDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/snli_ve/instruct_defaults.yaml"}