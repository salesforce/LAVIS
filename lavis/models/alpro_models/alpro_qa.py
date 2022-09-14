"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from warnings import warn

import torch
import torch.nn.functional as F
from lavis.common.config import node_to_dict
from lavis.common.registry import registry
from lavis.models.alpro_models import AlproBase
from lavis.models.alpro_models.alpro_outputs import (
    AlproIntermediateOutput,
    AlproOutputWithLogits,
)
from lavis.models.med import XBertEncoder
from lavis.models.timesformer.vit import TimeSformer
from torch import nn


@registry.register_model("alpro_qa")
class AlproQA(AlproBase):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "msrvtt": "configs/models/alpro_qa_msrvtt.yaml",
        "msvd": "configs/models/alpro_qa_msvd.yaml",
    }

    def __init__(
        self, visual_encoder, text_encoder, hidden_size, num_classes, max_txt_len=40
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()

        self.visual_encoder = visual_encoder

        self.text_encoder = text_encoder

        if num_classes > 0:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 2),
                nn.ReLU(True),
                nn.Linear(hidden_size * 2, num_classes),
            )
        else:
            warn(f"num_classes is 0. Initialized {type(self)} without classifier.")

        self.max_txt_len = max_txt_len

    def forward(self, samples, is_train=True):
        visual_inputs = samples["video"]
        question = samples["text_input"]
        targets = samples["answers"]

        # forward text
        text = self.tokenizer(
            question,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(self.device)

        text_output = self.text_encoder.forward_text(
            text,
            token_type_ids=torch.zeros(
                text.input_ids.shape, dtype=torch.long, device=self.device
            ),
        )
        text_embeds = text_output.last_hidden_state

        # forward visual
        # timeSformer asks for (b, c, t, h, w) as input.
        video_embeds = self.visual_encoder.forward_features(visual_inputs)
        video_atts = torch.ones(video_embeds.size()[:-1], dtype=torch.long).to(
            self.device
        )

        # forward cross-encoder
        attention_mask = torch.cat([text.attention_mask, video_atts], dim=1)
        embedding_output = torch.cat([text_embeds, video_embeds], dim=1)

        encoder_output = self.text_encoder(
            encoder_embeds=embedding_output,
            attention_mask=attention_mask,
            return_dict=True,
            mode="fusion",
        )

        prediction = self.classifier(encoder_output.last_hidden_state[:, 0, :])
        if is_train:
            loss = F.cross_entropy(prediction, targets)
            # return {"loss": loss}
            return AlproOutputWithLogits(
                loss=loss,
                intermediate_output=AlproIntermediateOutput(
                    video_embeds=video_embeds,
                    text_embeds=text_embeds,
                    encoder_output=encoder_output,
                ),
                logits=prediction,
            )
        else:
            return {"predictions": prediction, "targets": targets}

    def predict(self, samples):
        output = self.forward(samples, is_train=False)
        return output

    @classmethod
    def from_config(cls, cfg):
        # vision encoder
        visual_encoder_config = node_to_dict(cfg.timesformer)
        visual_encoder = TimeSformer(**visual_encoder_config)

        # text encoder
        text_encoder = XBertEncoder.from_config(cfg)

        num_classes = cfg.get("num_classes", -1)
        hidden_size = cfg.get("hidden_size", 768)

        model = cls(
            visual_encoder=visual_encoder,
            text_encoder=text_encoder,
            hidden_size=hidden_size,
            num_classes=num_classes,
        )

        num_patches = (
            visual_encoder_config["image_size"] // visual_encoder_config["patch_size"]
        ) ** 2
        num_frames = visual_encoder_config["n_frms"]

        model.load_checkpoint_from_config(
            cfg, num_frames=num_frames, num_patches=num_patches
        )

        return model
