"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import torch
import torch.nn as nn
from lavis.common.registry import registry
from lavis.models.base_model import BaseModel
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import GPT2LMHeadModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions


@registry.register_model("gpt_dialogue")
class GPTDialogue(BaseModel, GPT2LMHeadModel):

    PRETRAINED_MODEL_CONFIG_DICT = {"base": "configs/models/gpt_dialogue_base.yaml"}

    def __init__(self, config, len_video_ft=4224):

        super().__init__(config)

        self.video_ff = nn.Linear(len_video_ft, config.n_embd)
        self.video_ff_out = nn.Linear(config.n_embd, len_video_ft)

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        samples,
        past_key_values=None,
        position_ids=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        input_embs = self.transformer.wte(samples["input_ids"])
        video_embs = self.video_ff(samples["video_fts"])
        input_embs = torch.cat([video_embs, input_embs], dim=1)

        transformer_outputs = self.transformer(
            attention_mask=samples["attn_mask"],
            token_type_ids=samples["token_type_ids"],
            inputs_embeds=input_embs,
            position_ids=position_ids,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if samples["labels"] is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = samples["labels"][..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

        if samples["video_fts"] is not None:
            len_video_fts = samples["video_fts"].shape[1]
            video_logits = self.video_ff_out(hidden_states[:, :len_video_fts, :])
            # Shift so that tokens < n predict n
            shift_logits = video_logits[..., :-1, :].contiguous()
            shift_labels = samples["video_fts"][..., 1:, :].contiguous()
            # Flatten the tokens
            loss_fct = MSELoss(reduction="mean")
            video_loss = loss_fct(shift_logits, shift_labels)

            if loss is not None:
                loss = loss + video_loss
            else:
                loss = video_loss

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    @classmethod
    def from_config(cls, cfg):
        model = cls.__bases__[1].from_pretrained("gpt2")
        model.resize_token_embeddings(cfg["len_tokenizer"])
        return model
