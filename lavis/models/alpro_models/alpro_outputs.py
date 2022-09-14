"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from dataclasses import dataclass
from typing import Optional

import torch
from transformers.modeling_outputs import (
    BaseModelOutputWithPoolingAndCrossAttentions,
    ModelOutput,
)


@dataclass
class AlproSimilarity(ModelOutput):
    sim_v2t: torch.FloatTensor = None
    sim_t2v: torch.FloatTensor = None

    sim_v2t_targets: Optional[torch.FloatTensor] = None
    sim_t2v_targets: Optional[torch.FloatTensor] = None


@dataclass
class AlproIntermediateOutput(ModelOutput):
    # uni-modal features
    video_embeds: torch.FloatTensor = None
    text_embeds: Optional[torch.FloatTensor] = None

    # intermediate outputs of multimodal encoder
    encoder_output: Optional[BaseModelOutputWithPoolingAndCrossAttentions] = None
    encoder_output_neg: Optional[BaseModelOutputWithPoolingAndCrossAttentions] = None

    vtm_logits: Optional[torch.FloatTensor] = None
    vtm_labels: Optional[torch.LongTensor] = None


@dataclass
class AlproOutput(ModelOutput):
    # some finetuned models (e.g. BlipVQA) do not compute similarity, thus optional.
    sims: Optional[AlproSimilarity] = None

    intermediate_output: AlproIntermediateOutput = None

    loss: Optional[torch.FloatTensor] = None

    loss_vtc: Optional[torch.FloatTensor] = None

    loss_vtm: Optional[torch.FloatTensor] = None

    loss_mlm: Optional[torch.FloatTensor] = None


@dataclass
class AlproOutputWithLogits(AlproOutput):
    logits: torch.FloatTensor = None
