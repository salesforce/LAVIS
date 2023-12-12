"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.common.registry import registry
from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder, MultiModalDatasetBuilder
from lavis.datasets.datasets.nlvr_datasets import NLVRDataset, NLVREvalDataset
from lavis.datasets.datasets.snli_ve_datasets import SNLIVisualEntialmentDataset, SNLIVisualEntialmentInstructDataset
from lavis.datasets.datasets.violin_dataset import ViolinVideoEntailmentDataset, ViolinVideoEntailmentInstructDataset
from lavis.datasets.datasets.vsr_datasets import VSRClassificationDataset, VSRClassificationInstructDataset
from lavis.datasets.datasets.audio_classification_datasets import ESC50
@registry.register_builder("violin_entailment")
class ViolinEntailmentBuilder(BaseDatasetBuilder):
    train_dataset_cls = ViolinVideoEntailmentDataset
    eval_dataset_cls = ViolinVideoEntailmentDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/violin/defaults_entail.yaml",
    }


@registry.register_builder("violin_entailment_instruct")
class ViolinEntailmentInstructBuilder(BaseDatasetBuilder):
    train_dataset_cls = ViolinVideoEntailmentInstructDataset
    eval_dataset_cls = ViolinVideoEntailmentInstructDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/violin/defaults_entail_instruct.yaml",
    }

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
class SNLIVisualEntailmentInstructBuilder(BaseDatasetBuilder):
    train_dataset_cls = SNLIVisualEntialmentInstructDataset
    eval_dataset_cls = SNLIVisualEntialmentInstructDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/snli_ve/defaults_instruct.yaml"}


@registry.register_builder("vsr_classification")
class VSRClassificationBuilder(BaseDatasetBuilder):
    train_dataset_cls = VSRClassificationDataset
    eval_dataset_cls = VSRClassificationDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/vsr/defaults_classification.yaml"}

@registry.register_builder("vsr_classification_instruct")
class SNLIVisualEntailmentInstructBuilder(BaseDatasetBuilder):
    train_dataset_cls = VSRClassificationInstructDataset
    eval_dataset_cls = VSRClassificationInstructDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/vsr/defaults_classification_instruct.yaml"}

@registry.register_builder("esc50_cls")
class ESC50ClassificationBuilder(MultiModalDatasetBuilder):
    eval_dataset_cls = ESC50

    DATASET_CONFIG_DICT = {"default": "configs/datasets/esc50/defaults_mm_cls.yaml"}
