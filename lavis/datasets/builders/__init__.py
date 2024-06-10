"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.datasets.builders.base_dataset_builder import load_dataset_config
from lavis.datasets.builders.caption_builder import (
    COCOCapBuilder,
    MSRVTTCapBuilder,
    MSVDCapBuilder,
    VATEXCapBuilder,
    MSRVTTCapInstructBuilder,
    MSVDCapInstructBuilder,
    VATEXCapInstructBuilder,
    WebVid2MCapBuilder,
    WebVid2MCapInstructBuilder,
    VALORCaptionBuilder,
    VALORCaptionInstructBuilder,
    ViolinCapBuilder,
    ViolinCapInstructBuilder,
    VlepCaptionInstructBuilder, 
    VlepCaptionBuilder,
    YouCookCaptionBuilder,
    YouCookCaptionInstructBuilder,
    COINCaptionBuilder,
    COINCaptionInstructBuilder,
    CharadeCaptionBuilder,
    CharadeCaptionInstructBuilder,
    TextCapsCapBuilder,
    TextCapsCapInstructBuilder,
    Flickr30kCapBuilder,
    Flickr30kCapInstructBuilder

)
from lavis.datasets.builders.image_text_pair_builder import (
    ConceptualCaption12MBuilder,
    ConceptualCaption12MInstructBuilder,
    ConceptualCaption3MBuilder,
    ConceptualCaption3MInstructBuilder,
    VGCaptionBuilder,
    VGCaptionInstructBuilder,
    SBUCaptionBuilder,
    SBUCaptionInstructBuilder,
    Laion400MBuilder,
    Laion400MInstructBuilder
)
from lavis.datasets.builders.classification_builder import (
    NLVRBuilder,
    SNLIVisualEntailmentBuilder,
    SNLIVisualEntailmentInstructBuilder,
    ViolinEntailmentInstructBuilder,
    ViolinEntailmentBuilder,
    ESC50ClassificationBuilder
)
from lavis.datasets.builders.imagefolder_builder import ImageNetBuilder
from lavis.datasets.builders.video_qa_builder import (
    MSRVTTQABuilder, 
    MSVDQABuilder,
    MSRVTTQAInstructBuilder,
    MSVDQAInstructBuilder,
    MusicAVQABuilder,
    MusicAVQAInstructBuilder
)

from lavis.datasets.builders.vqa_builder import (
    COCOVQABuilder,
    COCOVQAInstructBuilder,
    OKVQABuilder,
    OKVQAInstructBuilder,
    AOKVQABuilder,
    AOKVQAInstructBuilder,
    VGVQABuilder,
    VGVQAInstructBuilder,
    GQABuilder,
    GQAInstructBuilder,
    IconQABuilder,
    IconQAInstructBuilder,
    ScienceQABuilder,
    ScienceQAInstructBuilder,
    OCRVQABuilder,
    OCRVQAInstructBuilder,
    VizWizVQABuilder
)
from lavis.datasets.builders.retrieval_builder import (
    MSRVTTRetrievalBuilder,
    DiDeMoRetrievalBuilder,
    COCORetrievalBuilder,
    Flickr30kBuilder,
)

from lavis.datasets.builders.audio_caption_builder import (
    AudioSetBuilder,
    AudioCapsCapBuilder,
    AudioSetInstructBuilder,
    AudioCapsInstructCapBuilder,
    WavCapsCapInstructBuilder,
    WavCapsCapBuilder
)

from lavis.datasets.builders.object3d_caption_builder import (
    ObjaverseCaptionInstructBuilder,
    ShapenetCaptionInstructBuilder,
    ObjaverseCaptionBuilder,
    ShapenetCaptionBuilder
)
from lavis.datasets.builders.object3d_qa_builder import ObjaverseQABuilder
from lavis.datasets.builders.object3d_classification_builder import ModelNetClassificationBuilder

from lavis.datasets.builders.audio_qa_builder import AudioCapsQABuilder, ClothoQABuilder

from lavis.datasets.builders.dialogue_builder import (
    AVSDDialBuilder, 
    AVSDDialInstructBuilder,
    YT8MDialBuilder,
    LLaVA150kDialInstructBuilder,
    VisDialBuilder,
    VisDialInstructBuilder
)
from lavis.datasets.builders.text_to_image_generation_builder import BlipDiffusionFinetuneBuilder

from lavis.datasets.builders.discrn_builders import DiscrnImagePcBuilder, DiscrnAudioVideoBuilder

from lavis.common.registry import registry

__all__ = [
    "BlipDiffusionFinetuneBuilder",
    "COCOCapBuilder",
    "COCORetrievalBuilder",
    "COCOVQABuilder",
    "ConceptualCaption12MBuilder",
    "ConceptualCaption3MBuilder",
    "DiDeMoRetrievalBuilder",
    "Flickr30kBuilder",
    "GQABuilder",
    "ImageNetBuilder",
    "MSRVTTCapBuilder",
    "MSRVTTQABuilder",
    "MSRVTTRetrievalBuilder",
    "MSVDCapBuilder",
    "MSVDQABuilder",
    "NLVRBuilder",
    "OKVQABuilder",
    "AOKVQABuilder",
    "SBUCaptionBuilder",
    "SNLIVisualEntailmentBuilder",
    "VATEXCapBuilder",
    "VGCaptionBuilder",
    "VGVQABuilder",
    "AVSDDialBuilder",
    "Laion400MBuilder",

    "ViolinCapBuilder",
    "ViolinEntailmentBuilder",
    "VlepCaptionBuilder",
    "YouCookCaptionBuilder",
    "COINCaptionBuilder",
    "CharadeCaptionBuilder",
    "YT8MDialBuilder",
    "IconQABuilder",
    "ScienceQABuilder",
    "VisDialBuilder",
    "OCRVQABuilder",
    "VizWizVQABuilder",
    "TextCapsCapBuilder",
    "Flickr30kCapBuilder",
    "AudioSetBuilder",
    "AudioCapsCapBuilder",
    "WavCapsCapBuilder",
    "WebVid2MCapBuilder",
    "VALORCaptionBuilder",
    "ObjaverseCaptionBuilder",
    "ShapenetCaptionBuilder",
    "ObjaverseQABuilder",
    "MusicAVQABuilder",
    "ESC50ClassificationBuilder",

    ## Instruction Builders
    "AOKVQAInstructBuilder",
    "OKVQAInstructBuilder",
    "AudioSetInstructBuilder",
    "AudioCapsInstructCapBuilder",
    "AudioCapsQABuilder",
    "WavCapsCapInstructBuilder",
    "ObjaverseCaptionInstructBuilder",
    "ShapenetCaptionInstructBuilder",
    "ModelNetClassificationBuilder",
    "ObjaverseCaptionInstructBuilder",
    "MSRVTTCapInstructBuilder",
    "MSVDCapInstructBuilder",
    "VATEXCapInstructBuilder",
    "WebVid2MCapInstructBuilder",
    "MSRVTTQAInstructBuilder",
    "MSVDQAInstructBuilder",
    "VALORCaptionInstructBuilder",
    "AVSDDialInstructBuilder",
    "VisDialInstructBuilder",
    "MusicAVQAInstructBuilder",
    "ViolinCapInstructBuilder",
    "ViolinEntailmentInstructBuilder",
    "VlepCaptionInstructBuilder", 
    "YouCookCaptionInstructBuilder",
    "COINCaptionInstructBuilder",
    "CharadeCaptionInstructBuilder",
    "COCOVQAInstructBuilder",
    "VGVQAInstructBuilder",
    "GQAInstructBuilder",
    "IconQAInstructBuilder",
    "SNLIVisualEntailmentInstructBuilder",
    "Laion400MInstructBuilder",
    "LLaVA150kDialInstructBuilder",
    "ScienceQAInstructBuilder",
    "OCRVQAInstructBuilder",
    "TextCapsCapInstructBuilder",
    "Flickr30kCapInstructBuilder",
    "ConceptualCaption12MInstructBuilder",
    "ConceptualCaption3MInstructBuilder",
    "VGCaptionInstructBuilder",
    "SBUCaptionInstructBuilder",
    "ClothoQABuilder",

    # DisCRN
    "DiscrnImagePcBuilder",
    "DiscrnAudioVideoBuilder"

]


def load_dataset(name, cfg_path=None, vis_path=None, data_type=None):
    """
    Example

    >>> dataset = load_dataset("coco_caption", cfg=None)
    >>> splits = dataset.keys()
    >>> print([len(dataset[split]) for split in splits])

    """
    if cfg_path is None:
        cfg = None
    else:
        cfg = load_dataset_config(cfg_path)

    try:
        builder = registry.get_builder_class(name)(cfg)
    except TypeError:
        print(
            f"Dataset {name} not found. Available datasets:\n"
            + ", ".join([str(k) for k in dataset_zoo.get_names()])
        )
        exit(1)

    if vis_path is not None:
        if data_type is None:
            # use default data type in the config
            data_type = builder.config.data_type

        assert (
            data_type in builder.config.build_info
        ), f"Invalid data_type {data_type} for {name}."

        builder.config.build_info.get(data_type).storage = vis_path

    dataset = builder.build_datasets()
    return dataset


class DatasetZoo:
    def __init__(self) -> None:
        self.dataset_zoo = {
            k: list(v.DATASET_CONFIG_DICT.keys())
            for k, v in sorted(registry.mapping["builder_name_mapping"].items())
        }

    def get_names(self):
        return list(self.dataset_zoo.keys())


dataset_zoo = DatasetZoo()
