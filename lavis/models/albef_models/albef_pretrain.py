"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from lavis.common.registry import registry
from lavis.common.utils import get_abs_path
from lavis.models.albef_models import AlbefBase
from lavis.models.albef_models.albef_outputs import (
    AlbefIntermediateOutput,
    AlbefOutput,
    AlbefSimilarity,
)
from lavis.models.base_model import MomentumDistilationMixin, SharedQueueMixin
from lavis.models.med import BertForMaskedLM
from lavis.models.vit import VisionTransformerEncoder
from torch import nn
from transformers import BertConfig


@registry.register_model("albef_pretrain")
class AlbefPretrain(AlbefBase, MomentumDistilationMixin, SharedQueueMixin):
    """
    ALBEF pretrain model.

    Supported model types:
        - base: ALBEF base model used for pretraining.
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "base": "configs/models/albef_pretrain_base.yaml",
    }

    def __init__(
        self,
        image_encoder,
        text_encoder,
        queue_size,
        embed_dim=256,
        mlm_mask_prob=0.15,
        temp=0.07,
        momentum=0.995,
        alpha=0.4,
        max_txt_len=30,
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()

        self.visual_encoder = image_encoder
        self.text_encoder = text_encoder

        text_width = text_encoder.config.hidden_size
        vision_width = image_encoder.vision_width

        self.embed_dim = embed_dim

        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)

        self.itm_head = nn.Linear(text_width, 2)

        # create the momentum encoder
        self.visual_encoder_m = deepcopy(self.visual_encoder)
        self.text_encoder_m = deepcopy(self.text_encoder)

        self.vision_proj_m = deepcopy(self.vision_proj)
        self.text_proj_m = deepcopy(self.text_proj)

        self.model_pairs = [
            [self.visual_encoder, self.visual_encoder_m],
            [self.text_encoder, self.text_encoder_m],
            [self.vision_proj, self.vision_proj_m],
            [self.text_proj, self.text_proj_m],
        ]
        self.copy_params()

        # create the queue
        self.register_buffer("image_queue", torch.randn(embed_dim, queue_size))
        self.register_buffer("text_queue", torch.randn(embed_dim, queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)

        self.queue_size = queue_size
        self.momentum = momentum
        self.temp = nn.Parameter(temp * torch.ones([]))

        self.alpha = alpha
        self.max_txt_len = max_txt_len

        self.mlm_probability = mlm_mask_prob

    def _rampup_factor(self, epoch, iters, num_iters_per_epoch):
        return min(1, (epoch * num_iters_per_epoch + iters) / (2 * num_iters_per_epoch))

    def forward(self, samples):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W). The input images. Default: H=224, W=224.
                - text_input (list): A list of length batch_size, each element is a string of text/caption.
                - epoch (int): The current epoch.
                - iters (int): The current iteration.
                - num_iters_per_epoch (int): The number of iterations per epoch.

        Returns:
            BlipOutput: A BlipOutput object containing loss and intermediate output. See ``lavis.models.blip_models.blip_outputs.BlipOutput`` for more details.

        Examples:
            >>> import torch
            >>> from lavis.models import load_model
            >>> model = load_model("albef_pretrain")
            >>> images = torch.randn(4, 3, 224, 224)
            >>> text_input = ["caption of image 1", "another caption of image 1", "caption of image 2", "caption of image 3"]
            >>> samples = {"image": images, "text_input": text_input, "epoch": 0, "iters": 0, "num_iters_per_epoch": 100}
            >>> output = model(samples)
            >>> output.keys()
            odict_keys(['sims', 'intermediate_output', 'loss', 'loss_itc', 'loss_itm', 'loss_mlm'])
        """
        image = samples["image"]
        caption = samples["text_input"]

        alpha = self.alpha * self._rampup_factor(
            epoch=samples["epoch"],
            iters=samples["iters"],
            num_iters_per_epoch=samples["num_iters_per_epoch"],
        )

        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)

        image_embeds = self.visual_encoder.forward_features(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            self.device
        )

        text = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(self.device)

        image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)

        text_output = self.text_encoder.bert(
            text.input_ids,
            attention_mask=text.attention_mask,
            return_dict=True,
            mode="text",
        )
        text_embeds = text_output.last_hidden_state
        text_feat = F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)

        # get momentum features
        with torch.no_grad():
            self._momentum_update()
            image_embeds_m = self.visual_encoder_m(image)
            image_feat_m = F.normalize(
                self.vision_proj_m(image_embeds_m[:, 0, :]), dim=-1
            )
            image_feat_all = torch.cat(
                [image_feat_m.t(), self.image_queue.clone().detach()], dim=1
            )
            text_output_m = self.text_encoder_m.bert(
                text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True,
                mode="text",
            )
            text_embeds_m = text_output_m.last_hidden_state
            text_feat_m = F.normalize(self.text_proj_m(text_embeds_m[:, 0, :]), dim=-1)
            text_feat_all = torch.cat(
                [text_feat_m.t(), self.text_queue.clone().detach()], dim=1
            )

            sim_i2t_m = image_feat_m @ text_feat_all / self.temp
            sim_t2i_m = text_feat_m @ image_feat_all / self.temp

            sim_targets = torch.zeros(sim_i2t_m.size()).to(image.device)
            sim_targets.fill_diagonal_(1)

            sim_i2t_targets = (
                alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
            )
            sim_t2i_targets = (
                alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets
            )

        sim_i2t = image_feat @ text_feat_all / self.temp
        sim_t2i = text_feat @ image_feat_all / self.temp

        loss_i2t = -torch.sum(
            F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1
        ).mean()
        loss_t2i = -torch.sum(
            F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1
        ).mean()

        loss_itc = (loss_i2t + loss_t2i) / 2

        self._dequeue_and_enqueue(image_feat_m, text_feat_m)

        # forward the positve image-text pair
        encoder_output_pos = self.text_encoder.bert(
            encoder_embeds=text_embeds,
            attention_mask=text.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
            mode="fusion",
        )
        with torch.no_grad():
            bs = image.size(0)

            weights_i2t = sim_i2t[:, :bs].clone()
            weights_t2i = sim_t2i[:, :bs].clone()

            weights_i2t.fill_diagonal_(-np.Inf)
            weights_t2i.fill_diagonal_(-np.Inf)

            weights_i2t = F.softmax(weights_i2t, dim=1)
            weights_t2i = F.softmax(weights_t2i, dim=1)

        # select a negative image for each text
        image_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeds_neg.append(image_embeds[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

        # select a negative text for each image
        text_embeds_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_embeds_neg.append(text_embeds[neg_idx])
            text_atts_neg.append(text.attention_mask[neg_idx])
        text_embeds_neg = torch.stack(text_embeds_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        text_embeds_all = torch.cat([text_embeds, text_embeds_neg], dim=0)
        text_atts_all = torch.cat([text.attention_mask, text_atts_neg], dim=0)

        image_embeds_all = torch.cat([image_embeds_neg, image_embeds], dim=0)
        image_atts_all = torch.cat([image_atts, image_atts], dim=0)

        encoder_output_neg = self.text_encoder.bert(
            encoder_embeds=text_embeds_all,
            attention_mask=text_atts_all,
            encoder_hidden_states=image_embeds_all,
            encoder_attention_mask=image_atts_all,
            return_dict=True,
            mode="fusion",
        )

        vl_embeddings = torch.cat(
            [
                encoder_output_pos.last_hidden_state[:, 0, :],
                encoder_output_neg.last_hidden_state[:, 0, :],
            ],
            dim=0,
        )
        itm_logits = self.itm_head(vl_embeddings)

        itm_labels = torch.cat(
            [torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
            dim=0,
        ).to(self.device)
        loss_itm = F.cross_entropy(itm_logits, itm_labels)

        # MLM
        input_ids = text.input_ids.clone()
        labels = input_ids.clone()

        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        input_ids, labels = self.mask(
            input_ids,
            self.text_encoder.config.vocab_size,
            self.device,
            targets=labels,
            probability_matrix=probability_matrix,
        )

        with torch.no_grad():
            logits_m = self.text_encoder_m(
                input_ids,
                attention_mask=text.attention_mask,
                encoder_hidden_states=image_embeds_m,
                encoder_attention_mask=image_atts,
                return_dict=True,
                return_logits=True,
            )
        mlm_output = self.text_encoder(
            input_ids,
            attention_mask=text.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
            labels=labels,
            soft_labels=F.softmax(logits_m, dim=-1),
            alpha=alpha,
        )
        loss_mlm = mlm_output.loss

        return AlbefOutput(
            loss=loss_itc + loss_itm + loss_mlm,
            loss_itc=loss_itc,
            loss_itm=loss_itm,
            loss_mlm=loss_mlm,
            sims=AlbefSimilarity(
                sim_i2t=sim_i2t,
                sim_t2i=sim_t2i,
                sim_i2t_m=sim_i2t_m,
                sim_t2i_m=sim_t2i_m,
                sim_i2t_targets=sim_i2t_targets,
                sim_t2i_targets=sim_t2i_targets,
            ),
            intermediate_output=AlbefIntermediateOutput(
                image_embeds=image_embeds,
                image_embeds_m=image_embeds_m,
                text_embeds=text_embeds,
                text_embeds_m=text_embeds_m,
                encoder_output=encoder_output_pos,
                encoder_output_neg=encoder_output_neg,
                itm_logits=itm_logits,
                itm_labels=itm_labels,
            ),
        )

    def mask(
        self,
        input_ids,
        vocab_size,
        device,
        targets=None,
        masked_indices=None,
        probability_matrix=None,
    ):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        if masked_indices is None:
            masked_indices = torch.bernoulli(probability_matrix).bool()

        masked_indices[input_ids == self.tokenizer.pad_token_id] = False
        masked_indices[input_ids == self.tokenizer.cls_token_id] = False

        if targets is not None:
            targets[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = (
            torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        )
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = (
            torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long).to(
            device
        )
        input_ids[indices_random] = random_words[indices_random]
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged

        if targets is not None:
            return input_ids, targets
        else:
            return input_ids

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
        momentum = cfg.get("momentum", 0.995)
        alpha = cfg.get("alpha", 0.4)
        mlm_mask_prob = cfg.get("mlm_mask_prob", 0.15)
        temp = cfg.get("temp", 0.07)
        max_txt_len = cfg.get("max_txt_len", 30)
        queue_size = cfg.get("queue_size", 65536)

        model = cls(
            image_encoder=image_encoder,
            text_encoder=text_encoder,
            queue_size=queue_size,
            embed_dim=embed_dim,
            mlm_mask_prob=mlm_mask_prob,
            temp=temp,
            momentum=momentum,
            alpha=alpha,
            max_txt_len=max_txt_len,
        )

        return model
