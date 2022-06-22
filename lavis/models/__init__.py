from lavis.common.registry import registry

from lavis.models.base_model import BaseModel

from lavis.models.albef_models.albef_classification import AlbefClassification
from lavis.models.albef_models.albef_nlvr import AlbefNLVR
from lavis.models.albef_models.albef_pretrain import AlbefPretrain
from lavis.models.albef_models.albef_retrieval import AlbefRetrieval
from lavis.models.albef_models.albef_vqa import AlbefVQA
from lavis.models.alpro_models.alpro_qa import AlproQA
from lavis.models.alpro_models.alpro_retrieval import AlproRetrieval
from lavis.models.blip_models.blip import BlipBase, BlipITM
from lavis.models.blip_models.blip_caption import BlipCaption
from lavis.models.blip_models.blip_classification import BlipClassification
from lavis.models.blip_models.blip_pretrain import BlipPretrain
from lavis.models.blip_models.blip_retrieval import BlipRetrieval
from lavis.models.blip_models.blip_vqa import BlipVQA
from lavis.models.med import XBertLMHeadDecoder
from lavis.models.vit import VisionTransformerEncoder
from lavis.models.clip_models.model import CLIP


def load_model(name, model_type="base", is_eval=False):
    model = registry.get_model_class(name).build_default_model(model_type=model_type)

    if is_eval:
        model.eval()

    return model


__all__ = [
    "load_model",
    "AlbefClassification",
    "AlbefNLVR",
    "AlbefVQA",
    "AlbefPretrain",
    "AlbefRetrieval",
    "AlproQA",
    "AlproRetrieval",
    "BaseModel",
    "BlipBase",
    "BlipCaption",
    "BlipClassification",
    "BlipITM",
    "BlipPretrain",
    "BlipRetrieval",
    "BlipVQA",
    "CLIP",
    "VisionTransformerEncoder",
    "XBertLMHeadDecoder",
]
