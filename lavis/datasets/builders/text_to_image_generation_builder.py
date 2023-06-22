"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.common.registry import registry
from lavis.datasets.datasets.subject_driven_t2i_dataset import (
    SubjectDrivenTextToImageDataset,
)
from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder


@registry.register_builder("blip_diffusion_finetune")
class BlipDiffusionFinetuneBuilder(BaseDatasetBuilder):
    train_dataset_cls = SubjectDrivenTextToImageDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/blip_diffusion_datasets/defaults.yaml"
    }

    def _download_ann(self):
        pass

    def build(self):
        self.build_processors()

        build_info = self.config.build_info

        dataset = self.train_dataset_cls(
            image_dir=build_info.images.storage,
            subject_text=build_info.subject_text,
            inp_image_processor=self.kw_processors["inp_vis_processor"],
            tgt_image_processor=self.kw_processors["tgt_vis_processor"],
            txt_processor=self.text_processors["eval"],
        )

        return {"train": dataset}
