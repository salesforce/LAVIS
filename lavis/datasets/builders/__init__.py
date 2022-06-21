from lavis.datasets.builders.coco_cap_builder import COCOCapBuilder
from lavis.datasets.builders.coco_vqa_builder import COCOVQABuilder
from lavis.datasets.builders.conceptual_caption_builder import (
    ConceptualCaption12MBuilder,
    ConceptualCaption3MBuilder,
)
from lavis.datasets.builders.flickr30k_builder import Flickr30kBuilder
from lavis.datasets.builders.nlvr_builder import NLVRBuilder
from lavis.datasets.builders.sbu_caption_builder import SBUCaptionBuilder
from lavis.datasets.builders.snli_ve_builder import SNLIVisualEntailmentBuilder
from lavis.datasets.builders.vg_caption_builder import VGCaptionBuilder
from lavis.datasets.builders.vg_vqa_builder import VGVQABuilder
from lavis.datasets.builders.imagenet_builder import ImageNetBuilder
from lavis.datasets.builders.video_qa_builder import MSRVTTQABuilder, MSVDQABuilder
from lavis.datasets.builders.retrieval_builder import (
    MSRVTTRetrievalBuilder,
    DiDeMoRetrievalBuilder,
    COCORetrievalBuilder,
)

from lavis.common.registry import registry

__all__ = [
    "COCOCapBuilder",
    "COCORetrievalBuilder",
    "COCOVQABuilder",
    "ConceptualCaption3MBuilder",
    "ConceptualCaption12MBuilder",
    "DiDeMoRetrievalBuilder",
    "Flickr30kBuilder",
    "ImageNetBuilder",
    "MSRVTTRetrievalBuilder",
    "SBUCaptionBuilder",
    "SNLIVisualEntailmentBuilder",
    "VGCaptionBuilder",
    "VGVQABuilder",
    "NLVRBuilder",
    "MSRVTTQABuilder",
    "MSVDQABuilder",
]


def load_dataset(name, cfg=None):
    """
    Example

    >>> dataset = load_dataset("coco_caption", cfg=None)
    >>> splits = dataset.keys()
    >>> print([len(dataset[split]) for split in splits])

    """
    builder = registry.get_builder_class(name)(cfg)
    dataset = builder.build_datasets()

    return dataset
