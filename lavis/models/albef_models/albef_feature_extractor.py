"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import warnings

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

        self.temp = nn.Parameter(0.07 * torch.ones([]))

    @torch.no_grad()
    def extract_features(self, samples, mode="multimodal"):
        """
        Extract features for multimodal or unimodal samples.

        Args:
            samples (dict): A dictionary of samples, containing the following keys:
                - image (torch.Tensor): A tensor of shape (B, C, H, W) containing the image.
                    Raw images should be preprocessed before being passed to feature extractor.
                - text_input (list): A list of strings containing the text, length B.
            mode (str): The mode of feature extraction. Can be either "multimodal", "text" or "image".
                If "multimodal", return image features and multimodal features;
                if "text", return text features;
                if "image", return image features.
                Default: "multimodal".

        Returns:
            An AlbefOutputFeatures object, see lavis/models/albef_models/albef_outputs.py for details.

        Examples:
        ```python
            >>> from PIL import Image
            >>> from lavis.models import load_model_and_preprocess
            >>> raw_image = Image.open("docs/data/merlion.png").convert("RGB")
            >>> caption = "a large fountain spewing water into the air"
            >>> model, vis_processors, txt_processors = load_model_and_preprocess("albef_feature_extractor", is_eval=True)
            >>> image = vis_processors["eval"](raw_image).unsqueeze(0)
            >>> text_input = txt_processors["eval"](caption)

            >>> sample = {"image": image, "text_input": [text_input]}

            >>> features_multimodal = model.extract_features(sample)
            >>> features_multimodal.keys()
            odict_keys(['image_embeds', 'multimodal_embeds'])
            >>> features_multimodal.image_embeds.shape
            torch.Size([1, 197, 768])
            >>> features_multimodal.multimodal_embeds.shape
            torch.Size([1, 12, 768])

            >>> features_text = model.extract_features(sample, mode="text")
            >>> features_text.keys()
            odict_keys(['text_embeds', 'text_features'])
            >>> features_text.text_embeds.shape
            torch.Size([1, 12, 768])
            >>> features_text.text_features.shape
            torch.Size([1, 12, 256])

            >>> features_image = model.extract_features(sample, mode="image")
            >>> features_image.keys()
            odict_keys(['image_embeds', 'image_features'])
            >>> features_image.image_embeds.shape
            torch.Size([1, 197, 768])
            >>> features_image.image_features.shape
            torch.Size([1, 197, 256])
        ```
        """
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
            image_features = F.normalize(self.vision_proj(image_embeds), dim=-1)

        if "text" in mode or "multimodal" in mode:
            assert (
                caption is not None
            ), "text must be provided if mode is 'text' or 'multimodal'"

            text = self.tokenizer(
                caption,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            text_output = self.text_encoder.bert(
                text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True,
                mode="text",
            )
            text_embeds = text_output.last_hidden_state
            text_features = F.normalize(self.text_proj(text_embeds), dim=-1)

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
            image_embeds_proj=image_features,
            text_embeds=text_embeds,
            text_embeds_proj=text_features,
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

        model = cls(
            image_encoder=image_encoder,
            text_encoder=text_encoder,
            embed_dim=embed_dim,
            max_txt_len=max_txt_len,
        )

        # load pre-trained weights
        pretrain_path = cfg.get("pretrained", None)
        if pretrain_path is not None:
            msg = model.load_from_pretrained(
                url_or_filename=pretrain_path, rename_text_keys=False
            )
        else:
            warnings.warn("No pretrained weights are loaded.")

        return model
