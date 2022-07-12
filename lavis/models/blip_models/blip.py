import torch
import torch.nn.functional as F
from lavis.common.utils import get_abs_path
from lavis.models.base_model import BaseModel
from lavis.models.blip_models import init_tokenizer, load_from_pretrained
from lavis.models.med import BertModel
from lavis.models.vit import VisionTransformer, VisionTransformerEncoder
from torch import nn
from transformers import BertConfig


class BlipBase(BaseModel):
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

        self.tokenizer = init_tokenizer()
        med_config = BertConfig.from_json_file(get_abs_path(med_config))
        med_config.encoder_width = vision_width
        self.text_encoder = BertModel(config=med_config, add_pooling_layer=False)

        embed_dim = 256
        text_width = vision_width

        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)

        if pretrained:
            self, msg = load_from_pretrained(self, pretrained)
            assert len(msg.missing_keys) == 0

    def forward(self, image, caption, mode, normalized=False):

        assert mode in [
            "image",
            "text",
            "multimodal",
        ], "mode parameter must be image, text, or multimodal"

        text = self.tokenizer(caption, return_tensors="pt", padding=True).to(
            self.device
        )

        if mode == "image":
            # return image features
            image_embeds = self.visual_encoder.forward_features(image)
            if normalized:
                image_embeds = self.vision_proj(image_embeds)
            return image_embeds

        elif mode == "text":
            # return text features
            text_output = self.text_encoder(
                text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True,
                mode="text",
            )
            text_embeds = text_output.last_hidden_state[:, 0, :]
            if normalized:
                text_embeds = self.text_proj(text_embeds)
            # return text_output.last_hidden_state
            return text_embeds

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
            return output.last_hidden_state


class BlipITM(BaseModel):
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

        self.tokenizer = init_tokenizer()
        med_config.encoder_width = vision_width
        self.text_encoder = BertModel(config=med_config, add_pooling_layer=False)

        text_width = self.text_encoder.config.hidden_size

        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)

        self.itm_head = nn.Linear(text_width, 2)

        if pretrained:
            self, msg = load_from_pretrained(self, pretrained)
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
