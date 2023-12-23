"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder

from lavis.common.registry import registry
from lavis.datasets.datasets.aok_vqa_datasets import AOKVQADataset, AOKVQAEvalDataset, AOKVQAInstructDataset
from lavis.datasets.datasets.coco_vqa_datasets import COCOVQADataset, COCOVQAEvalDataset, COCOVQAInstructDataset
from lavis.datasets.datasets.vg_vqa_datasets import VGVQADataset, VGVQAInstructDataset
from lavis.datasets.datasets.gqa_datasets import GQADataset, GQAEvalDataset, GQAInstructDataset
from lavis.datasets.datasets.iconqa_datasets import IconQADataset, IconQAEvalDataset, IconQAInstructDataset
from lavis.datasets.datasets.ocr_datasets import OCRVQADataset, OCRVQAInstructDataset
from lavis.datasets.datasets.vizwiz_vqa_datasets import VizWizEvalDataset

@registry.register_builder("coco_vqa")
class COCOVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = COCOVQADataset
    eval_dataset_cls = COCOVQAEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/coco/defaults_vqa.yaml",
        "eval": "configs/datasets/coco/eval_vqa.yaml",
    }

@registry.register_builder("coco_vqa_instruct")
class COCOVQAInstructBuilder(BaseDatasetBuilder):
    train_dataset_cls = COCOVQAInstructDataset
    eval_dataset_cls = COCOVQAEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/coco/defaults_vqa_instruct.yaml",
        "eval": "configs/datasets/coco/eval_vqa.yaml",
    }

@registry.register_builder("vg_vqa")
class VGVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = VGVQADataset
    DATASET_CONFIG_DICT = {"default": "configs/datasets/vg/defaults_vqa.yaml"}

@registry.register_builder("vg_vqa_instruct")
class VGVQAInstructBuilder(BaseDatasetBuilder):
    train_dataset_cls = VGVQAInstructDataset
    DATASET_CONFIG_DICT = {"default": "configs/datasets/vg/defaults_vqa_instruct.yaml"}

@registry.register_builder("ok_vqa")
class OKVQABuilder(COCOVQABuilder):
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/okvqa/defaults.yaml",
    }

@registry.register_builder("ok_vqa_instruct")
class OKVQAInstructBuilder(COCOVQAInstructBuilder):
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/okvqa/defaults_instruct.yaml",
    }

@registry.register_builder("aok_vqa")
class AOKVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = AOKVQADataset
    eval_dataset_cls = AOKVQAEvalDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/aokvqa/defaults.yaml"}

@registry.register_builder("aok_vqa_instruct")
class AOKVQAInstructBuilder(BaseDatasetBuilder):
    train_dataset_cls = AOKVQAInstructDataset
    eval_dataset_cls = AOKVQAEvalDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/aokvqa/defaults_instruct.yaml"}


@registry.register_builder("gqa")
class GQABuilder(BaseDatasetBuilder):
    train_dataset_cls = GQADataset
    eval_dataset_cls = GQAEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/gqa/defaults.yaml",
        "balanced_val": "configs/datasets/gqa/balanced_val.yaml",
        "balanced_testdev": "configs/datasets/gqa/balanced_testdev.yaml",
    }

@registry.register_builder("gqa_instruct")
class GQAInstructBuilder(BaseDatasetBuilder):
    train_dataset_cls = GQAInstructDataset
    eval_dataset_cls = GQAEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/gqa/defaults_instruct.yaml",
        "balanced_val": "configs/datasets/gqa/balanced_val_instruct.yaml",
        "balanced_testdev": "configs/datasets/gqa/balanced_testdev_instruct.yaml",
    }

@registry.register_builder("iconqa")
class IconQABuilder(BaseDatasetBuilder):
    train_dataset_cls = IconQADataset
    eval_dataset_cls = IconQAEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/iconqa/defaults.yaml",
    }

@registry.register_builder("iconqa_instruct")
class IconQAInstructBuilder(BaseDatasetBuilder):
    train_dataset_cls = IconQAInstructDataset
    eval_dataset_cls = IconQAEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/iconqa/defaults_instruct.yaml",
    }

@registry.register_builder("scienceqa")
class ScienceQABuilder(BaseDatasetBuilder):
    train_dataset_cls = IconQADataset
    eval_dataset_cls = IconQAEvalDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/scienceqa/defaults.yaml"}

@registry.register_builder("scienceqa_instruct")
class ScienceQAInstructBuilder(BaseDatasetBuilder):
    train_dataset_cls = IconQAInstructDataset
    eval_dataset_cls = IconQAEvalDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/scienceqa/defaults_instruct.yaml"}

@registry.register_builder("ocr_vqa")
class OCRVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = OCRVQADataset
    eval_dataset_cls = OCRVQADataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/ocrvqa/defaults.yaml"}

@registry.register_builder("ocr_vqa_instruct")
class OCRVQAInstructBuilder(BaseDatasetBuilder):
    train_dataset_cls = OCRVQAInstructDataset
    eval_dataset_cls = OCRVQAInstructDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/ocrvqa/defaults_instruct.yaml"}


@registry.register_builder("vizwiz_vqa")
class VizWizVQABuilder(BaseDatasetBuilder):
    eval_dataset_cls = VizWizEvalDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/vizwiz/defaults.yaml"}



