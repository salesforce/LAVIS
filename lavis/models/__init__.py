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


def get_model_config(model_name, model_type="base"):
    import json

    from omegaconf import OmegaConf

    config_path = registry.get_model_class(model_name).PRETRAINED_MODEL_DICT[model_type]

    config = OmegaConf.load(config_path)
    config = OmegaConf.to_container(config)

    print(json.dumps(config, indent=4, sort_keys=True))

    return config


def load_model(name, model_type="base", is_eval=False, device="cpu"):
    model = registry.get_model_class(name).from_pretrained(model_type=model_type)

    if is_eval:
        model.eval()

    return model.to(device)


class ModelZoo:
    def __init__(self) -> None:
        self.model_zoo = {
            k: list(v.PRETRAINED_MODEL_DICT.keys())
            for k, v in registry.mapping["model_name_mapping"].items()
        }

    def __repr__(self) -> str:
        return (
            "=" * 50
            + "\n"
            + f"{'Architectures':<30} {'Types'}\n"
            + "=" * 50
            + "\n"
            + "\n".join(
                [
                    f"{name:<30} {', '.join(types)}"
                    for name, types in self.model_zoo.items()
                ]
            )
        )

    def __iter__(self):
        return iter(self.model_zoo.items())


model_zoo = ModelZoo()
