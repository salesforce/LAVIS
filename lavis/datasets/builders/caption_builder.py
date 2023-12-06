"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder, MultiModalDatasetBuilder
from lavis.datasets.datasets.capfilt_dataset import CapFiltCaptionInstructDataset, CapFiltCaptionDataset
from lavis.datasets.datasets.coco_caption_datasets import (
    COCOCapDataset,
    COCOCapInstructDataset,
    COCOCapEvalDataset,
    NoCapsEvalDataset,
)

from lavis.common.registry import registry
from lavis.datasets.datasets.video_caption_datasets import (
    VideoCaptionDataset,
    VideoCaptionEvalDataset,
    ClipCaptionDataset,
    ClipCaptionInstructDataset,
    ClipCaptionEvalDataset,
    VideoCaptionInstructDataset,
    WebVideoCaptionDataset,
    WebVideoCaptionInstructDataset,
)
from lavis.datasets.datasets.violin_dataset import (
    ViolinVideoCaptionDataset,
    ViolinVideoCaptionInstructDataset,
    ViolinVideoCaptionEvalDataset
)
from lavis.datasets.datasets.valor_caption import VALORCaptionInstuctDataset, VALORCaptionEvalDataset, VALORCaptionDataset
from lavis.datasets.datasets.vatex_captioning_datasets import VATEXCaptionInstuctDataset, VATEXCaptionEvalDataset, VATEXCaptionDataset
from lavis.datasets.datasets.vlep_dataset import VlepVideoDataset, VlepVideoInstructDataset, VlepVideoEvalDataset
from lavis.datasets.datasets.vsr_datasets import VSRCaptionDataset, VSRCaptionInstructDataset, VSRCaptionEvalDataset
from lavis.datasets.datasets.textcaps_datasets import TextCapsCapDataset, TextCapsCapInstructDataset, TextCapsCapEvalDataset

@registry.register_builder("coco_caption")
class COCOCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = COCOCapDataset
    eval_dataset_cls = COCOCapEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/coco/defaults_cap.yaml",
    }

@registry.register_builder("coco_caption_instruct")
class COCOCapInstructBuilder(BaseDatasetBuilder):
    train_dataset_cls = COCOCapInstructDataset
    eval_dataset_cls = COCOCapEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/coco/defaults_cap_instruct.yaml",
    }


@registry.register_builder("flickr30k_caption")
class Flickr30kCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = COCOCapDataset
    eval_dataset_cls = COCOCapEvalDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/flickr30k/defaults_cap.yaml",
    }

@registry.register_builder("flickr30k_caption_instruct")
class Flickr30kCapInstructBuilder(BaseDatasetBuilder):
    train_dataset_cls = COCOCapInstructDataset
    eval_dataset_cls = COCOCapEvalDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/flickr30k/defaults_cap_instuct.yaml",
    }

@registry.register_builder("nocaps")
class COCOCapBuilder(BaseDatasetBuilder):
    eval_dataset_cls = NoCapsEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/nocaps/defaults.yaml",
    }

@registry.register_builder("vsr_caption")
class VSRCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = VSRCaptionDataset
    eval_dataset_cls = VSRCaptionEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/vsr/defaults.yaml",
    }

@registry.register_builder("vsr_caption_instruct")
class VSRCapInstructBuilder(BaseDatasetBuilder):
    train_dataset_cls = VSRCaptionInstructDataset
    eval_dataset_cls = VSRCaptionEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/vsr/defaults.yaml",
    }

@registry.register_builder("textcaps_caption")
class TextCapsCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = TextCapsCapDataset
    eval_dataset_cls = TextCapsCapEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/textcaps/defaults.yaml",
    }

@registry.register_builder("textcaps_caption_instruct")
class TextCapsCapInstructBuilder(BaseDatasetBuilder):
    train_dataset_cls = TextCapsCapInstructDataset
    eval_dataset_cls = TextCapsCapEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/textcaps/defaults_instruct.yaml",
    }


@registry.register_builder("capfilt14m")
class CapFiltCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = CapFiltCaptionDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/capfilt14m/defaults_cap.yaml",
    }

@registry.register_builder("capfilt14m_instruct")
class CapFiltCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = CapFiltCaptionInstructDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/capfilt14m/defaults_cap_instruct.yaml",
    }


@registry.register_builder("msrvtt_caption")
class MSRVTTCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = VideoCaptionDataset
    eval_dataset_cls = VideoCaptionEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/msrvtt/defaults_cap.yaml",
    }


@registry.register_builder("msvd_caption")
class MSVDCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = VideoCaptionDataset
    eval_dataset_cls = VideoCaptionEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/msvd/defaults_cap.yaml",
    }


@registry.register_builder("vatex_caption")
class VATEXCapBuilder(MultiModalDatasetBuilder):
    train_dataset_cls = VATEXCaptionDataset
    eval_dataset_cls = VATEXCaptionEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/vatex/defaults_cap.yaml",
    }

@registry.register_builder("msrvtt_caption_instruct")
class MSRVTTCapInstructBuilder(BaseDatasetBuilder):
    train_dataset_cls = VideoCaptionInstructDataset
    eval_dataset_cls = VideoCaptionEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/msrvtt/defaults_cap_instruct.yaml",
    }

@registry.register_builder("msvd_caption_instruct")
class MSVDCapInstructBuilder(BaseDatasetBuilder):
    train_dataset_cls = VideoCaptionInstructDataset
    eval_dataset_cls = VideoCaptionEvalDataset


    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/msvd/defaults_cap_instruct.yaml",
    }


@registry.register_builder("vatex_caption_instruct")
class VATEXCapInstructBuilder(MultiModalDatasetBuilder):
    train_dataset_cls = VATEXCaptionInstuctDataset
    eval_dataset_cls = VATEXCaptionEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/vatex/defaults_cap_instruct.yaml",
    }


@registry.register_builder("webvid2m_caption")
class WebVid2MCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = WebVideoCaptionDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/webvid/defaults_cap.yaml",
    }

@registry.register_builder("webvid2m_caption_instruct")
class WebVid2MCapInstructBuilder(BaseDatasetBuilder):
    train_dataset_cls = WebVideoCaptionInstructDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/webvid/defaults_cap_instruct.yaml",
    }

@registry.register_builder("violin_caption")
class ViolinCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = ViolinVideoCaptionDataset
    eval_dataset_cls = ViolinVideoCaptionEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/violin/defaults_cap.yaml",
    }


@registry.register_builder("violin_caption_instruct")
class ViolinCapInstructBuilder(BaseDatasetBuilder):
    train_dataset_cls = ViolinVideoCaptionInstructDataset
    eval_dataset_cls = ViolinVideoCaptionEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/violin/defaults_cap_instruct.yaml",
    }

@registry.register_builder("valor_mm_caption")
class VALORCaptionBuilder(MultiModalDatasetBuilder):
    train_dataset_cls = VALORCaptionDataset
    eval_dataset_cls = VALORCaptionEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/valor/defaults_mm_cap.yaml"
    }

@registry.register_builder("valor_mm_caption_instruct")
class VALORCaptionInstructBuilder(MultiModalDatasetBuilder):
    train_dataset_cls = VALORCaptionInstuctDataset
    eval_dataset_cls = VALORCaptionEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/valor/defaults_mm_cap_instruct.yaml"
    }

@registry.register_builder("vlep_caption")
class VlepCaptionBuilder(BaseDatasetBuilder):
    train_dataset_cls = VlepVideoDataset
    eval_dataset_cls = VlepVideoEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/vlep/defaults_cap.yaml"
    }


@registry.register_builder("vlep_caption_instruct")
class VlepCaptionInstructBuilder(BaseDatasetBuilder):
    train_dataset_cls = VlepVideoInstructDataset
    eval_dataset_cls = VlepVideoEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/vlep/defaults_cap_instruct.yaml"
    }

@registry.register_builder("youcook_caption")
class YouCookCaptionBuilder(BaseDatasetBuilder):
    train_dataset_cls = ClipCaptionDataset
    eval_dataset_cls = ClipCaptionEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/youcook/defaults_cap.yaml",
    }

@registry.register_builder("youcook_caption_instruct")
class YouCookCaptionInstructBuilder(BaseDatasetBuilder):
    train_dataset_cls = ClipCaptionInstructDataset
    eval_dataset_cls = ClipCaptionEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/youcook/defaults_cap_instruct.yaml",
    }

@registry.register_builder("coin_caption")
class COINCaptionBuilder(BaseDatasetBuilder):
    train_dataset_cls = ClipCaptionDataset
    eval_dataset_cls = ClipCaptionEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/coin/defaults_cap.yaml",
    }


@registry.register_builder("coin_caption_instruct")
class COINCaptionInstructBuilder(BaseDatasetBuilder):
    train_dataset_cls = ClipCaptionInstructDataset
    eval_dataset_cls = ClipCaptionEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/coin/defaults_cap_instruct.yaml",
    }


@registry.register_builder("charade_caption")
class CharadeCaptionBuilder(BaseDatasetBuilder):
    train_dataset_cls = ClipCaptionDataset
    eval_dataset_cls = ClipCaptionEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/charade/defaults_cap.yaml",
    }

@registry.register_builder("charade_caption_instruct")
class CharadeCaptionInstructBuilder(BaseDatasetBuilder):
    train_dataset_cls = ClipCaptionInstructDataset
    eval_dataset_cls = ClipCaptionEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/charade/defaults_cap_instruct.yaml",
    }
