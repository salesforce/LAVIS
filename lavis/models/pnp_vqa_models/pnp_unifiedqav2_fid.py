"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

 Based on facebookresearch code base
 https://github.com/facebookresearch/FiD
"""

import torch
import torch.nn as nn
from lavis.common.registry import registry
from lavis.models.base_model import BaseModel
from lavis.common.utils import get_abs_path
from transformers import T5Config, T5Tokenizer, T5ForConditionalGeneration


@registry.register_model("pnp_unifiedqav2_fid")
class PNPUnifiedQAv2FiD(T5ForConditionalGeneration, BaseModel):

    PRETRAINED_MODEL_CONFIG_DICT = {}

    def __init__(self, config, model_path):
        super().__init__(config)
        
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        if input_ids != None:
            if input_ids.dim() == 3:
                self.encoder.num_contexts = input_ids.size(1)
            input_ids = input_ids.view(input_ids.size(0), -1)
        if attention_mask != None:
            attention_mask = attention_mask.view(attention_mask.size(0), -1)

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )

    def generate(self, input_ids, attention_mask, num_beams=1, min_length=0, max_length=20):
        self.encoder.num_contexts = input_ids.size(1)

        return super().generate(
            input_ids=input_ids.view(input_ids.size(0), -1),
            attention_mask=attention_mask.view(attention_mask.size(0), -1),
            num_beams=num_beams,
            min_length=min_length,
            max_length=max_length
        )

    def load_unifiedqa(self, state_dict):
        self.load_state_dict(state_dict)
        self.encoder = T5EncoderWrapper(self.encoder)

    @classmethod
    def from_config(cls, cfg):
        model_path = cfg.get('pretrained')
        t5_config_path = get_abs_path(cfg.get("t5_config_path"))
        t5_config = T5Config.from_json_file(t5_config_path)
        model = cls(t5_config, model_path)
        model.load_unifiedqa(T5ForConditionalGeneration.from_pretrained(model_path).state_dict())

        return model


class T5EncoderWrapper(torch.nn.Module):

    def __init__(self, encoder):
        super().__init__()

        self.encoder = encoder
        self.block = self.encoder.block
        self.parallelize = self.encoder.parallelize
        self.main_input_name = encoder.main_input_name

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        bsz, total_length = input_ids.shape
        context_length = total_length // self.num_contexts
        input_ids = input_ids.view(bsz*self.num_contexts, context_length)
        attention_mask = attention_mask.view(bsz*self.num_contexts, context_length)
        outputs = self.encoder(input_ids, attention_mask, **kwargs)
        outputs = (outputs[0].view(bsz, self.num_contexts*context_length, -1), ) + outputs[1:]

        return outputs