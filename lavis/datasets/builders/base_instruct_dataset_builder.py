"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
import os
import shutil
import warnings

import lavis.common.utils as utils
import torch.distributed as dist
from lavis.common.dist_utils import is_dist_avail_and_initialized, is_main_process
from lavis.common.registry import registry
from lavis.datasets.data_utils import extract_archive
from lavis.processors.base_processor import BaseProcessor
from omegaconf import OmegaConf
from torchvision.datasets.utils import download_url

from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder

class BaseInstructDatasetBuilder(BaseDatasetBuilder):
    # train_dataset_cls, eval_dataset_cls = None, None

    def __init__(self, cfg=None):
        super().__init__(cfg)

        assert hasattr(self.config, "instruction_path"), "For instruction tuning dataset, config.instruction_path is required!"

    def build(self):
        """
        Create by split datasets inheriting torch.utils.data.Datasets.

        # build() can be dataset-specific. Overwrite to customize.
        """
        self.build_processors()

        build_info = self.config.build_info

        ann_info = build_info.annotations
        vis_info = build_info.get(self.data_type)

        datasets = dict()
        for split in ann_info.keys():
            if split not in ["train", "val", "test"]:
                continue

            is_train = split == "train"

            # processors
            vis_processor = (
                self.vis_processors["train"]
                if is_train
                else self.vis_processors["eval"]
            )
            text_processor = (
                self.text_processors["train"]
                if is_train
                else self.text_processors["eval"]
            )

            # annotation path
            ann_paths = ann_info.get(split).storage
            if isinstance(ann_paths, str):
                ann_paths = [ann_paths]

            abs_ann_paths = []
            for ann_path in ann_paths:
                if not os.path.isabs(ann_path):
                    ann_path = utils.get_cache_path(ann_path)
                abs_ann_paths.append(ann_path)
            ann_paths = abs_ann_paths

            # visual data storage path
            vis_path = vis_info.storage

            if not os.path.isabs(vis_path):
                # vis_path = os.path.join(utils.get_cache_path(), vis_path)
                vis_path = utils.get_cache_path(vis_path)

            if not os.path.exists(vis_path):
                warnings.warn("storage path {} does not exist.".format(vis_path))

            # create datasets
            if is_train:
                datasets[split] = self.train_dataset_cls(
                    vis_processor=vis_processor,
                    text_processor=text_processor,
                    ann_paths=ann_paths,
                    vis_root=vis_path,
                    instruction_path=self.config.instruction_path,
                )
            else:
                datasets[split] = self.eval_dataset_cls(
                    vis_processor=vis_processor,
                    text_processor=text_processor,
                    ann_paths=ann_paths,
                    vis_root=vis_path,
                )


        return datasets


def load_dataset_config(cfg_path):
    cfg = OmegaConf.load(cfg_path).datasets
    cfg = cfg[list(cfg.keys())[0]]

    return cfg
