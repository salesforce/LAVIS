from lavis.processors.base_processor import BaseProcessor

from lavis.processors.alpro_processors import (
    AlproVideoTrainProcessor,
    AlproVideoEvalProcessor,
)
from lavis.processors.blip_processors import (
    BlipImageTrainProcessor,
    BlipImageEvalProcessor,
    BlipCaptionProcessor,
)
from lavis.processors.blipv2_processors import (
    BlipV2ImageBaseProcessor,
    BlipV2ImageTrainProcessor,
    BlipV2ImageEvalProcessor,
    BlipV2QuestionProcessor,
)
from lavis.processors.clip_processors import ClipImageTrainProcessor

from lavis.common.registry import registry

__all__ = [
    "BaseProcessor",
    # ALPRO
    "AlproVideoTrainProcessor",
    "AlproVideoEvalProcessor",
    # BLIP
    "BlipImageTrainProcessor",
    "BlipImageEvalProcessor",
    "BlipCaptionProcessor",
    # BLIP_V2
    "BlipV2ImageBaseProcessor",
    "BlipV2ImageTrainProcessor",
    "BlipV2ImageEvalProcessor",
    "BlipV2QuestionProcessor",
    "ClipImageTrainProcessor",
]


def load_processor(name, cfg=None):
    """
    Example

    >>> processor = load_processor("alpro_video_train", cfg=None)
    """
    processor = registry.get_processor_class(name).from_config(cfg)

    return processor
