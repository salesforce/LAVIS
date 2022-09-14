"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.common.registry import registry
from lavis.processors.blip_processors import BlipImageBaseProcessor
from omegaconf import OmegaConf
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode


def _convert_to_rgb(image):
    return image.convert("RGB")


@registry.register_processor("clip_image_train")
class ClipImageTrainProcessor(BlipImageBaseProcessor):
    def __init__(
        self, image_size=224, mean=None, std=None, min_scale=0.9, max_scale=1.0
    ):

        super().__init__(mean=mean, std=std)

        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(min_scale, max_scale),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                _convert_to_rgb,
                transforms.ToTensor(),
                self.normalize,
            ]
        )

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", 224)

        mean = cfg.get("mean", None)
        std = cfg.get("std", None)

        min_scale = cfg.get("min_scale", 0.9)
        max_scale = cfg.get("max_scale", 1.0)

        return cls(
            image_size=image_size,
            mean=mean,
            std=std,
            min_scale=min_scale,
            max_scale=max_scale,
        )


@registry.register_processor("clip_image_eval")
class ClipImageEvalProcessor(BlipImageBaseProcessor):
    def __init__(self, image_size=224, mean=None, std=None):

        super().__init__(mean=mean, std=std)

        self.transform = transforms.Compose(
            [
                transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop(image_size),
                _convert_to_rgb,
                transforms.ToTensor(),
                self.normalize,
            ]
        )

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", 224)

        mean = cfg.get("mean", None)
        std = cfg.get("std", None)

        return cls(
            image_size=image_size,
            mean=mean,
            std=std,
        )
