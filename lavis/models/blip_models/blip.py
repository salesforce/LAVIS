import logging
import os

import torch
import torch.nn.functional as F
from lavis.common.utils import get_abs_path, is_url
from lavis.models.base_model import BaseModel
from lavis.models.blip_models.blip_outputs import BlipOutputFeatures
from lavis.models.med import BertModel
from lavis.models.vit import VisionTransformerEncoder, interpolate_pos_embed
from lavis.processors import load_processor
from timm.models.hub import download_cached_file
from torch import nn
from transformers import BertConfig, BertTokenizer


class BlipBase(BaseModel):
    @classmethod
    def init_tokenizer(cls):
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        tokenizer.add_special_tokens({"additional_special_tokens": ["[ENC]"]})
        tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]
        return tokenizer

    def load_from_pretrained(self, url_or_filename):
        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location="cpu")
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location="cpu")
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        state_dict = checkpoint["model"]

        state_dict["visual_encoder.pos_embed"] = interpolate_pos_embed(
            state_dict["visual_encoder.pos_embed"], self.visual_encoder
        )
        if "visual_encoder_m.pos_embed" in self.state_dict().keys():
            state_dict["visual_encoder_m.pos_embed"] = interpolate_pos_embed(
                state_dict["visual_encoder_m.pos_embed"], self.visual_encoder_m
            )

        for key in self.state_dict().keys():
            if key in state_dict.keys():
                if state_dict[key].shape != self.state_dict()[key].shape:
                    del state_dict[key]

        msg = self.load_state_dict(state_dict, strict=False)

        logging.info("Missing keys {}".format(msg.missing_keys))
        logging.info("load checkpoint from %s" % url_or_filename)

        return msg


class BlipFeatureExtractor(BlipBase):
    def __init__(
        self,
        med_config="configs/models/med_config.json",
        image_size=224,
        vit="base",
        vit_grad_ckpt=False,
        vit_ckpt_layer=0,
        pretrained="",
    ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """
        DeprecationWarning(
            "BlipFeatureExtractor is deprecated, use lavis.models.blip_models.blip_feature_extractor.BlipFeatureExtractor instead"
        )

        super().__init__()

        if vit == "base":
            vision_width = 768
            self.visual_encoder = VisionTransformerEncoder(
                img_size=image_size,
                patch_size=16,
                embed_dim=vision_width,
                depth=12,
                num_heads=12,
                use_grad_checkpointing=vit_grad_ckpt,
                ckpt_layer=vit_ckpt_layer,
                drop_path_rate=0,
            )
        else:
            raise NotImplementedError("")

        self.tokenizer = self.init_tokenizer()
        med_config = BertConfig.from_json_file(get_abs_path(med_config))
        med_config.encoder_width = vision_width
        self.text_encoder = BertModel(config=med_config, add_pooling_layer=False)

        embed_dim = 256
        text_width = vision_width

        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)

        self.temp = nn.Parameter(0.07 * torch.ones([]))

        if pretrained:
            msg = self.load_from_pretrained(pretrained)
            assert len(msg.missing_keys) == 0

    def forward(self, image=None, caption=None, mode="multimodal", normalized=True):

        assert mode in [
            "image",
            "text",
            "multimodal",
        ], "mode parameter must be image, text, or multimodal, but got {}".format(mode)

        text = self.tokenizer(caption, return_tensors="pt", padding=True).to(
            self.device
        )

        if mode == "image":
            # return image features
            image_embeds = self.visual_encoder.forward_features(image)

            image_features = self.vision_proj(image_embeds)
            if normalized:
                image_features = F.normalize(image_features, dim=-1)

            return BlipOutputFeatures(
                image_embeds=image_embeds, image_features=image_features
            )

        elif mode == "text":
            # return text features
            text_output = self.text_encoder(
                text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True,
                mode="text",
            )
            text_embeds = text_output.last_hidden_state

            text_features = self.text_proj(text_embeds)
            if normalized:
                text_features = F.normalize(text_features, dim=-1)

            return BlipOutputFeatures(
                text_embeds=text_embeds, text_features=text_features
            )

        elif mode == "multimodal":
            # return multimodel features
            image_embeds = self.visual_encoder.forward_features(image)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                self.device
            )

            text.input_ids[:, 0] = self.tokenizer.enc_token_id
            output = self.text_encoder(
                text.input_ids,
                attention_mask=text.attention_mask,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            multimodal_embeds = output.last_hidden_state

            return BlipOutputFeatures(multimodal_embeds=multimodal_embeds)


class BlipITM(BlipBase):
    def __init__(
        self,
        med_config="configs/models/med_config.json",
        image_size=384,
        vit="base",
        vit_grad_ckpt=False,
        vit_ckpt_layer=0,
        embed_dim=256,
        pretrained="",
        drop_path_rate=0,
    ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """
        super().__init__()

        med_config = BertConfig.from_json_file(get_abs_path(med_config))
        if vit == "base":
            vision_width = 768
            self.visual_encoder = VisionTransformerEncoder(
                img_size=image_size,
                patch_size=16,
                embed_dim=vision_width,
                depth=12,
                num_heads=12,
                use_grad_checkpointing=vit_grad_ckpt,
                ckpt_layer=vit_ckpt_layer,
                drop_path_rate=0,
            )
            med_config.encoder_width = vision_width
        elif vit == "large":
            vision_width = 1024
            self.visual_encoder = VisionTransformerEncoder(
                img_size=image_size,
                patch_size=16,
                embed_dim=vision_width,
                depth=24,
                num_heads=16,
                use_grad_checkpointing=vit_grad_ckpt,
                ckpt_layer=vit_ckpt_layer,
                drop_path_rate=drop_path_rate or 0.1,
            )
            med_config.encoder_width = vision_width

        self.tokenizer = self.init_tokenizer()
        med_config.encoder_width = vision_width
        self.text_encoder = BertModel(config=med_config, add_pooling_layer=False)

        text_width = self.text_encoder.config.hidden_size

        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)

        self.itm_head = nn.Linear(text_width, 2)

        if pretrained:
            msg = self.load_from_pretrained(pretrained)
            assert len(msg.missing_keys) == 0

    def forward(self, image, caption, match_head="itm"):

        image_embeds = self.visual_encoder.forward_features(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        text = self.tokenizer(
            caption,
            padding="longest",
            truncation=True,
            max_length=35,
            return_tensors="pt",
        ).to(image.device)
        if match_head == "itm":
            encoder_input_ids = text.input_ids.clone()
            encoder_input_ids[:, 0] = self.tokenizer.enc_token_id  # extra code
            output = self.text_encoder(
                encoder_input_ids,
                attention_mask=text.attention_mask,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            itm_output = self.itm_head(output.last_hidden_state[:, 0, :])
            return itm_output

        elif match_head == "itc":
            text_output = self.text_encoder(
                text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True,
                mode="text",
            )
            image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)
            text_feat = F.normalize(
                self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
            )

            sim = image_feat @ text_feat.t()
            return sim


def load_feature_extractor(device, model_path_or_url=None, prompt="", max_words=50):
    if model_path_or_url is None:
        model_path_or_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base.pth"

    model = BlipFeatureExtractor(pretrained=model_path_or_url)
    model.eval()
    model = model.to(device)

    vis_processor = load_processor("blip_image_eval").build(image_size=224)
    text_processor = load_processor("blip_caption").build(
        prompt=prompt, max_words=max_words
    )

    return model, vis_processor, text_processor
