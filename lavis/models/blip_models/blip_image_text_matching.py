"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import torch
import torch.nn.functional as F
from lavis.common.registry import registry
from lavis.models.blip_models.blip import BlipBase
from torch import nn
from lavis.models.med import XBertEncoder

from lavis.models.vit import VisionTransformerEncoder


@registry.register_model("blip_image_text_matching")
class BlipITM(BlipBase):
    """
    BLIP Image-Text Matching (ITM) model.

    Supported model types:
        - base: fine-tuned BLIP retrieval weights on COCO dataset (Karpathy split).
        - large: fine-tuned BLIP retrieval weights on COCO dataset (Karpathy split).

    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip_image_text_matching", "base")
        >>> model = load_model("blip_image_text_matching", "large")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "base": "configs/models/blip_itm_base.yaml",
        "large": "configs/models/blip_itm_large.yaml",
    }

    def __init__(self, image_encoder, text_encoder, embed_dim=256, max_txt_len=35):
        super().__init__()

        self.tokenizer = self.init_tokenizer()

        self.text_encoder = text_encoder

        self.visual_encoder = image_encoder

        self.max_txt_len = max_txt_len

        # creating projection layers for ITC
        text_width = text_encoder.config.hidden_size
        vision_width = image_encoder.vision_width

        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)

        self.itm_head = nn.Linear(text_width, 2)

    def forward(self, samples, match_head="itm"):
        image = samples["image"]
        caption = samples["text_input"]

        image_embeds = self.visual_encoder.forward_features(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        text = self.tokenizer(
            caption,
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
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
    def itm_rank(self, image_embeds, image_atts, encoder_input_ids, match_head='itm'):
        # breakpoint()
        encoder_input_ids = encoder_input_ids.clone()
        encoder_input_ids = encoder_input_ids[:, 3:]
        text_attention_mask = (encoder_input_ids != self.tokenizer.pad_token_id).long()

        if match_head == 'itm':
            # encoder_input_ids = encoder_input_ids.clone()
            encoder_input_ids[:, 0] = self.tokenizer.enc_token_id
            output = self.text_encoder(encoder_input_ids,
                                       attention_mask=text_attention_mask,
                                       encoder_hidden_states=image_embeds,
                                       encoder_attention_mask=image_atts,
                                       return_dict=True,
                                       )
            # print(output.last_hidden_state.shape)
            itm_output = self.itm_head(output.last_hidden_state[:, 0, :])
            itm_output = F.softmax(itm_output, dim=1)[:,1]
            return itm_output #, mask, token_length

        elif match_head == 'itc':
            encoder_input_ids[:, 0] = self.tokenizer.cls_token_id
            text_output = self.text_encoder(encoder_input_ids, attention_mask=text_attention_mask,
                                            return_dict=True, mode='text')
            image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)
            text_feat = F.normalize(self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1)

            sim = image_feat @ text_feat.t()
            return sim

    @classmethod
    def from_config(cls, cfg=None):
        image_encoder = VisionTransformerEncoder.from_config(cfg)
        text_encoder = XBertEncoder.from_config(cfg)

        embed_dim = cfg.get("embed_dim", 256)
        max_txt_len = cfg.get("max_txt_len", 35)

        model = cls(
            image_encoder=image_encoder,
            text_encoder=text_encoder,
            embed_dim=embed_dim,
            max_txt_len=max_txt_len,
        )

        model.load_checkpoint_from_config(cfg)

        return model


def compute_gradcam(model, visual_input, text_input, tokenized_text, block_num=6):
    model.text_encoder.base_model.base_model.encoder.layer[
        block_num
    ].crossattention.self.save_attention = True

    output = model({"image": visual_input, "text_input": text_input}, match_head="itm")
    loss = output[:, 1].sum()

    model.zero_grad()
    loss.backward()
    with torch.no_grad():
        mask = tokenized_text.attention_mask.view(
            tokenized_text.attention_mask.size(0), 1, -1, 1, 1
        )  # (bsz,1,token_len, 1,1)
        token_length = tokenized_text.attention_mask.sum(dim=-1) - 2
        token_length = token_length.cpu()
        # grads and cams [bsz, num_head, seq_len, image_patch]
        grads = model.text_encoder.base_model.base_model.encoder.layer[
            block_num
        ].crossattention.self.get_attn_gradients()
        cams = model.text_encoder.base_model.base_model.encoder.layer[
            block_num
        ].crossattention.self.get_attention_map()

        # assume using vit with 576 num image patch
        cams = cams[:, :, :, 1:].reshape(visual_input.size(0), 12, -1, 24, 24) * mask
        grads = (
            grads[:, :, :, 1:].clamp(0).reshape(visual_input.size(0), 12, -1, 24, 24)
            * mask
        )

        gradcams = cams * grads
        gradcam_list = []

        for ind in range(visual_input.size(0)):
            token_length_ = token_length[ind]
            gradcam = gradcams[ind].mean(0).cpu().detach()
            # [enc token gradcam, average gradcam across token, gradcam for individual token]
            gradcam = torch.cat(
                (
                    gradcam[0:1, :],
                    gradcam[1 : token_length_ + 1, :].sum(dim=0, keepdim=True)
                    / token_length_,
                    gradcam[1:, :],
                )
            )
            gradcam_list.append(gradcam)
            
    return gradcam_list, output
