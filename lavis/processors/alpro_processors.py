"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import torch
from lavis.common.registry import registry
from lavis.datasets.data_utils import load_video, load_clip
from lavis.processors import transforms_video
from lavis.processors.base_processor import BaseProcessor
from lavis.processors.randaugment import VideoRandomAugment
from lavis.processors import functional_video as F
from omegaconf import OmegaConf
from torchvision import transforms

MAX_INT = registry.get("MAX_INT")


class AlproVideoBaseProcessor(BaseProcessor):
    def __init__(self, mean=None, std=None, n_frms=MAX_INT):
        if mean is None:
            mean = (0.48145466, 0.4578275, 0.40821073)
        if std is None:
            std = (0.26862954, 0.26130258, 0.27577711)

        self.normalize = transforms_video.NormalizeVideo(mean, std)

        self.n_frms = n_frms


class ToUint8(object):
    def __init__(self):
        pass

    def __call__(self, tensor):
        return tensor.to(torch.uint8)

    def __repr__(self):
        return self.__class__.__name__


class ToTHWC(object):
    """
    Args:
        clip (torch.tensor, dtype=torch.uint8): Size is (C, T, H, W)
    Return:
        clip (torch.tensor, dtype=torch.float): Size is (T, H, W, C)
    """

    def __init__(self):
        pass

    def __call__(self, tensor):
        return tensor.permute(1, 2, 3, 0)

    def __repr__(self):
        return self.__class__.__name__


class ResizeVideo(object):
    def __init__(self, target_size, interpolation_mode="bilinear"):
        self.target_size = target_size
        self.interpolation_mode = interpolation_mode

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (C, T, H, W)
        Returns:
            torch.tensor: central cropping of video clip. Size is
            (C, T, crop_size, crop_size)
        """
        return F.resize(clip, self.target_size, self.interpolation_mode)

    def __repr__(self):
        return self.__class__.__name__ + "(resize_size={0})".format(self.target_size)


@registry.register_processor("alpro_video_train")
class AlproVideoTrainProcessor(AlproVideoBaseProcessor):
    def __init__(
        self,
        image_size=384,
        mean=None,
        std=None,
        min_scale=0.5,
        max_scale=1.0,
        n_frms=MAX_INT,
        full_video=True,
    ):
        super().__init__(mean=mean, std=std, n_frms=n_frms)

        self.image_size = image_size
        self.full_video=full_video

        self.transform = transforms.Compose(
            [
                # Video size is (C, T, H, W)
                transforms_video.RandomResizedCropVideo(
                    image_size,
                    scale=(min_scale, max_scale),
                    interpolation_mode="bicubic",
                ),
                transforms_video.RandomHorizontalFlipVideo(),
                ToTHWC(),  # C, T, H, W -> T, H, W, C
                VideoRandomAugment(
                    2,
                    5,
                    augs=[
                        "Identity",
                        # "AutoContrast",
                        "Brightness",
                        "Sharpness",
                        "Equalize",
                        "ShearX",
                        "ShearY",
                        "TranslateX",
                        "TranslateY",
                        "Rotate",
                    ],
                ),
                ToUint8(),
                transforms_video.ToTensorVideo(),  # T, H, W, C -> C, T, H, W
                self.normalize,
            ]
        )

    def __call__(self, vpath, start_sec=None, end_sec=None):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (C, T, H, W)
        Returns:
            torch.tensor: video clip after transforms. Size is (C, T, size, size).
        """
        if self.full_video:
            clip = load_video( ## initial LAVIS code has errors when loading video.
                video_path=vpath,
                n_frms=self.n_frms,
                height=self.image_size,
                width=self.image_size,
                sampling="headtail",
            )
            # clip = load_clip(
            #     video_path=vpath, 
            #     num_frames=self.n_frms, 
            #     target_height=self.image_size, 
            #     target_width=self.image_size,
            #     start_time=start_sec,
            #     end_time=end_sec, 
            #     sampling="headtail"
            #     )
        else:
            clip = load_clip(
                video_path=vpath, 
                num_frames=self.n_frms, 
                target_height=self.image_size, 
                target_width=self.image_size,
                start_time=start_sec,
                end_time=end_sec, 
                sampling="headtail"
                )
        transformed = self.transform(clip)

        ## repeat last frame for padding
        pad_size = self.n_frms - transformed.shape[1]
        if pad_size>0:
            last_frame = transformed[:, -1, :, :].unsqueeze(1)
            repeat_frames = last_frame.repeat(1, pad_size, 1, 1)
            transformed = torch.cat([transformed, repeat_frames], dim=1)

        return transformed

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", 256)

        mean = cfg.get("mean", None)
        std = cfg.get("std", None)

        min_scale = cfg.get("min_scale", 0.5)
        max_scale = cfg.get("max_scale", 1.0)

        n_frms = cfg.get("n_frms", MAX_INT)
        full_video = cfg.get("full_video", True)

        return cls(
            image_size=image_size,
            mean=mean,
            std=std,
            min_scale=min_scale,
            max_scale=max_scale,
            n_frms=n_frms,
            full_video=full_video
        )


@registry.register_processor("alpro_video_eval")
class AlproVideoEvalProcessor(AlproVideoBaseProcessor):
    def __init__(self, image_size=256, mean=None, std=None, n_frms=MAX_INT,  full_video=True):
        super().__init__(mean=mean, std=std, n_frms=n_frms)

        self.image_size = image_size
        self.full_video=full_video

        # Input video size is (C, T, H, W)
        self.transform = transforms.Compose(
            [
                # frames will be resized during decord loading.
                ToUint8(),  # C, T, H, W
                ToTHWC(),  # T, H, W, C
                transforms_video.ToTensorVideo(),  # C, T, H, W
                self.normalize,  # C, T, H, W
            ]
        )

    def __call__(self, vpath, start_sec=None, end_sec=None):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (C, T, H, W)
        Returns:
            torch.tensor: video clip after transforms. Size is (C, T, size, size).
        """
        if self.full_video:
            clip = load_clip(
                video_path=vpath, 
                num_frames=self.n_frms, 
                target_height=self.image_size, 
                target_width=self.image_size,
                start_time=start_sec,
                end_time=end_sec, 
                sampling="headtail"
                )
        else:
            clip = load_clip(
                video_path=vpath, 
                num_frames=self.n_frms, 
                target_height=self.image_size, 
                target_width=self.image_size,
                start_time=start_sec,
                end_time=end_sec, 
                sampling="headtail"
                )
        transformed = self.transform(clip)

        ## repeat last frame for padding
        pad_size = self.n_frms - transformed.shape[1]
        if pad_size>0:
            last_frame = transformed[:, -1, :, :].unsqueeze(1)
            repeat_frames = last_frame.repeat(1, pad_size, 1, 1)
            transformed = torch.cat([transformed, repeat_frames], dim=1)

        return transformed

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", 256)
        full_video = cfg.get("full_video", True)

        mean = cfg.get("mean", None)
        std = cfg.get("std", None)

        n_frms = cfg.get("n_frms", MAX_INT)

        return cls(image_size=image_size, mean=mean, std=std, n_frms=n_frms, full_video=full_video)
