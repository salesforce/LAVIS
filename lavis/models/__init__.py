import logging
from omegaconf import OmegaConf
from lavis.common.registry import registry

from lavis.models.base_model import BaseModel

from lavis.models.albef_models.albef_classification import AlbefClassification
from lavis.models.albef_models.albef_nlvr import AlbefNLVR
from lavis.models.albef_models.albef_pretrain import AlbefPretrain
from lavis.models.albef_models.albef_retrieval import AlbefRetrieval
from lavis.models.albef_models.albef_vqa import AlbefVQA
from lavis.models.alpro_models.alpro_qa import AlproQA
from lavis.models.alpro_models.alpro_retrieval import AlproRetrieval
from lavis.models.blip_models.blip import BlipBase, BlipFeatureExtractor, BlipITM
from lavis.models.blip_models.blip_caption import BlipCaption
from lavis.models.blip_models.blip_classification import BlipClassification
from lavis.models.blip_models.blip_nlvr import BlipNLVR
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
    "BlipFeatureExtractor",
    "BlipCaption",
    "BlipClassification",
    "BlipITM",
    "BlipNLVR",
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


def load_preprocess(config):
    def _build_proc_from_cfg(cfg):
        print(cfg)
        return (
            registry.get_processor_class(cfg.name).from_config(cfg)
            if cfg is not None
            else None
        )

    vis_processors = dict()
    txt_processors = dict()

    vis_proc_cfg = config.get("vis_processor")
    txt_proc_cfg = config.get("text_processor")

    if vis_proc_cfg is not None:
        vis_train_cfg = vis_proc_cfg.get("train")
        vis_eval_cfg = vis_proc_cfg.get("eval")

        vis_processors["train"] = _build_proc_from_cfg(vis_train_cfg)
        vis_processors["eval"] = _build_proc_from_cfg(vis_eval_cfg)

    if txt_proc_cfg is not None:
        txt_train_cfg = txt_proc_cfg.get("train")
        txt_eval_cfg = txt_proc_cfg.get("eval")

        txt_processors["train"] = _build_proc_from_cfg(txt_train_cfg)
        txt_processors["eval"] = _build_proc_from_cfg(txt_eval_cfg)

    return vis_processors, txt_processors


def load_model_and_preprocess(name, model_type="base", is_eval=False, device="cpu"):
    model_cls = registry.get_model_class(name)

    # load model
    model = model_cls.from_pretrained(model_type=model_type)

    if is_eval:
        model.eval()

    # load preprocess
    cfg = OmegaConf.load(model_cls.default_config_path(model_type))
    if cfg is not None:
        preprocess_cfg = cfg.preprocess

        vis_processors, txt_processors = load_preprocess(preprocess_cfg)
    else:
        vis_processors, txt_processors = None, None
        logging.info(
            f"""No default preprocess for model {name} ({model_type}).
                This can happen if the model is not finetuned on downstream datasets,
                or it is not intended for direct use without finetuning.
            """
        )

    return model.to(device), vis_processors, txt_processors


class ModelZoo:
    def __init__(self) -> None:
        self.model_zoo = {
            k: list(v.PRETRAINED_MODEL_DICT.keys())
            for k, v in registry.mapping["model_name_mapping"].items()
        }

    def __str__(self) -> str:
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

    def __len__(self):
        return sum([len(v) for v in self.model_zoo.values()])


model_zoo = ModelZoo()
