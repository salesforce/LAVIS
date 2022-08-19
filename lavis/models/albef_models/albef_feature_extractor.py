import torch
import torch.nn.functional as F
from lavis.common.registry import registry
from lavis.common.utils import get_abs_path
from lavis.models.albef_models import AlbefBase
from lavis.models.albef_models.albef_outputs import AlbefOutputFeatures
from lavis.models.med import BertForMaskedLM
from lavis.models.vit import VisionTransformerEncoder
from torch import nn
from transformers import BertConfig


@registry.register_model("albef_feature_extractor")
class AlbefFeatureExtractor(AlbefBase):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "base": "configs/models/albef_feature_extractor.yaml",
    }

    def __init__(self, image_encoder, text_encoder, embed_dim=256, max_txt_len=30):
        super().__init__()

        self.tokenizer = self.init_tokenizer()

        self.visual_encoder = image_encoder
        self.text_encoder = text_encoder

        text_width = text_encoder.config.hidden_size
        vision_width = image_encoder.vision_width

        self.embed_dim = embed_dim

        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)

        self.max_txt_len = max_txt_len

    @torch.no_grad()
    def extract_features(self, samples, mode="multimodal"):
        image = samples["image"]
        caption = samples["text_input"]

        if isinstance(mode, str):
            mode = [mode]

        for m in mode:
            assert m in [
                "multimodal",
                "image",
                "text",
            ], "mode must be one of [multimodal, image, text], but got {}".format(m)

        # initalize output
        image_embeds, text_embeds, multimodal_embeds = None, None, None
        image_features, text_features = None, None

        if "image" in mode or "multimodal" in mode:
            assert (
                image is not None
            ), "image must be provided if mode is 'image' or 'multimodal'"

            image_embeds = self.visual_encoder.forward_features(image)
            image_features = F.normalize(
                self.vision_proj(image_embeds[:, 0, :]), dim=-1
            )

        if "text" in mode or "multimodal" in mode:
            assert (
                caption is not None
            ), "text must be provided if mode is 'text' or 'multimodal'"

            text = self.tokenizer(
                caption,
                padding="max_length",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(self.device)

            text_output = self.text_encoder.bert(
                text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True,
                mode="text",
            )
            text_embeds = text_output.last_hidden_state
            text_features = F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)

        if "multimodal" in mode:
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                self.device
            )

            # forward the positve image-text pair
            output = self.text_encoder.bert(
                encoder_embeds=text_embeds,
                attention_mask=text.attention_mask,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
                mode="fusion",
            )

            multimodal_embeds = output.last_hidden_state

        return AlbefOutputFeatures(
            image_embeds=image_embeds,
            image_features=image_features,
            text_embeds=text_embeds,
            text_features=text_features,
            multimodal_embeds=multimodal_embeds,
        )

    @classmethod
    def from_config(cls, cfg=None):
        image_encoder = VisionTransformerEncoder.from_config(cfg, from_pretrained=True)
        config_text_encoder = BertConfig.from_json_file(
            get_abs_path(cfg["med_config_path"])
        )
        config_text_encoder.fusion_layer = 6
        text_encoder = BertForMaskedLM.from_pretrained(
            "bert-base-uncased", config=config_text_encoder
        )

        embed_dim = cfg.get("embed_dim", 256)
        max_txt_len = cfg.get("max_txt_len", 30)

        return cls(
            image_encoder=image_encoder,
            text_encoder=text_encoder,
            embed_dim=embed_dim,
            max_txt_len=max_txt_len,
        )
