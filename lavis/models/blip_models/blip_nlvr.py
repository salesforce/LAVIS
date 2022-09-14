"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os

import torch
import torch.nn.functional as F
from lavis.common.dist_utils import download_cached_file
from lavis.common.registry import registry
from lavis.common.utils import get_abs_path, is_url
from lavis.models.base_model import MomentumDistilationMixin
from lavis.models.blip_models.blip import BlipBase
from lavis.models.blip_models.blip_outputs import BlipIntermediateOutput, BlipOutput
from lavis.models.blip_models.nlvr_encoder import BertModel
from lavis.models.vit import VisionTransformerEncoder, interpolate_pos_embed
from torch import nn
from transformers import BertConfig


@registry.register_model("blip_nlvr")
class BlipNLVR(BlipBase, MomentumDistilationMixin):
    """
    Class for BLIP NLVR model.

    Supported model types:
        - base: model with pre-trained BLIP weights, used as initialization for fine-tuning.
        - nlvr: finetuned model on NLVR2 dataset.

    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip_nlvr", "nlvr")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "nlvr": "configs/models/blip_nlvr.yaml",
    }

    def __init__(self, image_encoder, text_encoder, num_classes):
        super().__init__()

        self.tokenizer = self.init_tokenizer()
        self.visual_encoder = image_encoder
        self.text_encoder = text_encoder

        hidden_size = text_encoder.config.hidden_size
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, samples, is_train=True):
        """
        Forward function for training and evaluation.

        Args:
            samples (dict): a dict of input samples, which contains the following keys:
                - image0 (torch.Tensor): input image 0, shape (batch_size, 3, H, W), default H=384, W=384.
                - image1 (torch.Tensor): input image 1, shape (batch_size, 3, H, W), default H=384, W=384.
                - text_input (list): list of strings, each string is a natural language sentence.
                - label (torch.LongTensor): ground truth label with shape (batch_size,).
            is_train (bool): whether the model is in training mode.
                If True, the model will return the loss;
                If False, the model will return the prediction.

        Examples:
            >>> import torch
            >>> from lavis.models import load_model
            >>> model = load_model("blip_nlvr", "nlvr")
            >>> samples = {
            ...     "image0": torch.randn(2, 3, 384, 384),
            ...     "image1": torch.randn(2, 3, 384, 384),
            ...     "text_input": ["there is a ferret in tall grass", "there are lips in one of the images"],
            ...     "label": torch.tensor([0, 1]),
            ... }
            >>> output = model(samples)
            >>> output.keys()
            odict_keys(['intermediate_output', 'loss'])
        """
        text = samples["text_input"]
        text = self.tokenizer(text, padding="longest", return_tensors="pt").to(
            self.device
        )
        text.input_ids[:, 0] = self.tokenizer.enc_token_id

        targets = samples["label"]

        image0 = samples["image0"]
        image1 = samples["image1"]
        images = torch.cat([image0, image1], dim=0)

        image_embeds = self.visual_encoder.forward_features(images)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            self.device
        )
        image0_embeds, image1_embeds = torch.split(image_embeds, targets.size(0))

        encoder_output = self.text_encoder(
            text.input_ids,
            attention_mask=text.attention_mask,
            encoder_hidden_states=[image0_embeds, image1_embeds],
            encoder_attention_mask=[
                image_atts[: image0_embeds.size(0)],
                image_atts[image0_embeds.size(0) :],
            ],
            return_dict=True,
        )

        prediction = self.cls_head(encoder_output.last_hidden_state[:, 0, :])

        if is_train:
            loss = F.cross_entropy(prediction, targets)
            # return {"loss": loss}
            return BlipOutput(
                loss=loss,
                intermediate_output=BlipIntermediateOutput(
                    image_embeds=torch.stack([image0_embeds, image1_embeds], dim=0),
                    encoder_output=encoder_output,
                ),
            )
        else:
            return {"predictions": prediction, "targets": targets}

    def predict(self, samples):
        output = self.forward(samples, is_train=False)
        return output

    @classmethod
    def from_config(cls, cfg=None):
        image_encoder = VisionTransformerEncoder.from_config(cfg)

        # text encoder + multimodal encoder
        bert_config = BertConfig.from_json_file(get_abs_path(cfg["med_config_path"]))
        text_encoder = BertModel(config=bert_config, add_pooling_layer=False)

        num_classes = cfg.get("num_classes", 3)

        assert num_classes > 1, "Invalid number of classes provided, found {}".format(
            num_classes
        )

        model = cls(
            image_encoder=image_encoder,
            text_encoder=text_encoder,
            num_classes=num_classes,
        )

        model.load_checkpoint_from_config(cfg)

        return model

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

        for key in list(state_dict.keys()):
            if "crossattention.self." in key:
                new_key0 = key.replace("self", "self0")
                new_key1 = key.replace("self", "self1")
                state_dict[new_key0] = state_dict[key]
                state_dict[new_key1] = state_dict[key]
            elif "crossattention.output.dense." in key:
                new_key0 = key.replace("dense", "dense0")
                new_key1 = key.replace("dense", "dense1")
                state_dict[new_key0] = state_dict[key]
                state_dict[new_key1] = state_dict[key]

        msg = self.load_state_dict(state_dict, strict=False)
        print("load checkpoint from %s" % url_or_filename)
        print(f"missing keys {msg.missing_keys}")
        return msg
