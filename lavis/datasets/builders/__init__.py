from lavis.datasets.builders.base_dataset_builder import load_dataset_config
from lavis.datasets.builders.coco_cap_builder import COCOCapBuilder
from lavis.datasets.builders.conceptual_caption_builder import (
    ConceptualCaption12MBuilder,
    ConceptualCaption3MBuilder,
)
from lavis.datasets.builders.nlvr_builder import NLVRBuilder
from lavis.datasets.builders.sbu_caption_builder import SBUCaptionBuilder
from lavis.datasets.builders.snli_ve_builder import SNLIVisualEntailmentBuilder
from lavis.datasets.builders.vg_caption_builder import VGCaptionBuilder
from lavis.datasets.builders.imagenet_builder import ImageNetBuilder
from lavis.datasets.builders.video_qa_builder import MSRVTTQABuilder, MSVDQABuilder
from lavis.datasets.builders.vqa_builder import (
    COCOVQABuilder,
    OKVQABuilder,
    VGVQABuilder,
)
from lavis.datasets.builders.retrieval_builder import (
    MSRVTTRetrievalBuilder,
    DiDeMoRetrievalBuilder,
    COCORetrievalBuilder,
    Flickr30kBuilder,
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
    "OKVQABuilder",
    "SBUCaptionBuilder",
    "SNLIVisualEntailmentBuilder",
    "VGCaptionBuilder",
    "VGVQABuilder",
    "NLVRBuilder",
    "MSRVTTQABuilder",
    "MSVDQABuilder",
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

    builder = registry.get_builder_class(name)(cfg)

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


def list_datasets():
    return {
        k: list(v.type2path.keys())
        for k, v in registry.mapping["builder_name_mapping"].items()
    }
