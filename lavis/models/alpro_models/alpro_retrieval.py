"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import datetime
import logging
import time

import lavis.common.dist_utils as dist_utils
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from lavis.common.config import node_to_dict
from lavis.common.dist_utils import get_rank
from lavis.common.logger import MetricLogger
from lavis.common.registry import registry
from lavis.models.alpro_models import AlproBase
from lavis.models.alpro_models.alpro_outputs import AlproIntermediateOutput, AlproOutput
from lavis.models.base_model import all_gather_with_grad
from lavis.models.med import XBertEncoder
from lavis.models.timesformer.vit import TimeSformer
from torch import nn


@registry.register_model("alpro_retrieval")
class AlproRetrieval(AlproBase):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "msrvtt": "configs/models/alpro_retrieval_msrvtt.yaml",
        "didemo": "configs/models/alpro_retrieval_didemo.yaml",
    }

    def __init__(
        self,
        visual_encoder,
        text_encoder,
        vision_width=768,
        text_width=768,
        embed_dim=256,
        max_txt_len=35,
        temp=0.07,
    ):
        super().__init__()

        self.temp = nn.Parameter(torch.ones([]) * temp)

        self.tokenizer = self.init_tokenizer()

        self.visual_encoder = visual_encoder
        self.text_encoder = text_encoder

        vision_width = vision_width
        text_width = text_width

        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)

        self.itm_head = nn.Linear(text_width, 2)

        self.max_txt_len = max_txt_len

    def forward(self, samples):
        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)

        visual_inputs = samples["video"]
        caption = samples["text_input"]

        b, t, c, h, w = visual_inputs.shape

        # forward text
        text = self.tokenizer(
            caption,
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
        text_feat = F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)

        # forward visual
        # timeSformer asks for (b, c, t, h, w) as input.
        video_embeds = self.visual_encoder.forward_features(visual_inputs)
        video_feat = F.normalize(self.vision_proj(video_embeds[:, 0, :]), dim=-1)
        video_atts = torch.ones(video_embeds.size()[:-1], dtype=torch.long).to(
            self.device
        )

        # ========== (in-batch) ITC loss ==========
        gathered_video_feats = all_gather_with_grad(video_feat)
        gathered_text_feats = all_gather_with_grad(text_feat)

        sim_v2t = video_feat @ gathered_text_feats.t() / self.temp
        sim_t2v = text_feat @ gathered_video_feats.t() / self.temp

        sim_targets = torch.zeros_like(sim_v2t)

        local_rank = get_rank()
        b_start, b_end = b * local_rank, b * (local_rank + 1)
        sim_targets[:, b_start:b_end] = torch.eye(b)

        loss_v2t = -torch.sum(F.log_softmax(sim_v2t, dim=1) * sim_targets, dim=1).mean()
        loss_t2v = -torch.sum(F.log_softmax(sim_t2v, dim=1) * sim_targets, dim=1).mean()

        vtc_loss = (loss_v2t + loss_t2v) / 2

        (
            vtm_loss,
            vtm_logits,
            vtm_labels,
            encoder_output,
            encoder_output_neg,
        ) = self.compute_vtm(
            text_embeds=text_embeds,
            text_atts=text.attention_mask,
            image_embeds=video_embeds,
            image_atts=video_atts,
            sim_i2t=sim_v2t.clone(),  # for hard mining
            sim_t2i=sim_t2v.clone(),  # for hard mining
        )

        loss = vtc_loss + vtm_loss

        # return {"loss": loss}
        return AlproOutput(
            loss=loss,
            loss_vtc=vtc_loss,
            loss_vtm=vtm_loss,
            intermediate_output=AlproIntermediateOutput(
                video_embeds=video_embeds,
                text_embeds=text_embeds,
                encoder_output=encoder_output,
                encoder_output_neg=encoder_output_neg,
                vtm_logits=vtm_logits,
                vtm_labels=vtm_labels,
            ),
        )

    def compute_vtm(
        self, text_embeds, text_atts, image_embeds, image_atts, sim_i2t, sim_t2i
    ):
        device = self.device

        # ====== positive pairs =======
        attention_mask = torch.cat([text_atts, image_atts], dim=1)
        embedding_output_pos = torch.cat([text_embeds, image_embeds], dim=1)

        encoder_outputs_pos = self.text_encoder(
            encoder_embeds=embedding_output_pos,
            attention_mask=attention_mask,
            return_dict=True,
            mode="fusion",
        )

        # ====== negative pairs =======
        bs = text_embeds.shape[0]

        local_rank = get_rank()
        b_start, b_end = bs * local_rank, bs * (local_rank + 1)

        with torch.no_grad():
            weights_v2t = sim_i2t[:, b_start:b_end]
            weights_t2v = sim_t2i[:, b_start:b_end]

            # never select self as negative
            weights_v2t.fill_diagonal_(-np.Inf)
            weights_t2v.fill_diagonal_(-np.Inf)

            weights_v2t = F.softmax(weights_v2t, dim=1)
            weights_t2v = F.softmax(weights_t2v, dim=1)

        # select a negative image for each text
        # FIXME to optimize using indexing operations
        image_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2v[b], 1).item()
            image_embeds_neg.append(image_embeds[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

        # select a negative text for each image
        text_embeds_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_v2t[b], 1).item()
            text_embeds_neg.append(text_embeds[neg_idx])
            text_atts_neg.append(text_atts[neg_idx])

        text_embeds_neg = torch.stack(text_embeds_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        text_embeds_all = torch.cat([text_embeds, text_embeds_neg], dim=0)
        text_atts_all = torch.cat([text_atts, text_atts_neg], dim=0)

        video_embeds_all = torch.cat([image_embeds_neg, image_embeds], dim=0)
        video_atts_all = torch.cat([image_atts, image_atts], dim=0)

        attention_mask_all = torch.cat([text_atts_all, video_atts_all], dim=1)
        embedding_output_all = torch.cat([text_embeds_all, video_embeds_all], dim=1)

        # forward negative pairs via cross encoder
        encoder_outputs_neg = self.text_encoder(
            encoder_embeds=embedding_output_all,
            attention_mask=attention_mask_all,
            return_dict=True,
            mode="fusion",
        )

        vl_embeddings = torch.cat(
            [
                encoder_outputs_pos.last_hidden_state[:, 0, :],
                encoder_outputs_neg.last_hidden_state[:, 0, :],
            ],
            dim=0,
        )
        vtm_logits = self.itm_head(vl_embeddings)

        vtm_labels = torch.cat(
            [torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
            dim=0,
        ).to(device)
        vtm_loss = F.cross_entropy(vtm_logits, vtm_labels)

        return (
            vtm_loss,
            vtm_logits,
            vtm_labels,
            encoder_outputs_pos,
            encoder_outputs_neg,
        )

    def compute_sim_matrix(self, data_loader, task_cfg):
        k_test = task_cfg.get("k_test")

        metric_logger = MetricLogger(delimiter="  ")
        header = "Evaluation:"

        logging.info("Computing features for evaluation...")
        start_time = time.time()

        texts = data_loader.dataset.text
        num_text = len(texts)
        text_bs = 256
        text_ids = []
        text_embeds = []
        text_feats = []
        text_atts = []
        for i in range(0, num_text, text_bs):
            text = texts[i : min(num_text, i + text_bs)]
            text_input = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(self.device)
            text_output = self.text_encoder.forward_text(
                text_input,
                token_type_ids=torch.zeros(
                    text_input.input_ids.shape, dtype=torch.long, device=self.device
                ),
            )
            text_feats.append(text_output.last_hidden_state.cpu())
            text_embed = F.normalize(
                self.text_proj(text_output.last_hidden_state[:, 0, :])
            )
            text_embeds.append(text_embed)
            text_ids.append(text_input.input_ids)
            text_atts.append(text_input.attention_mask)

        text_embeds = torch.cat(text_embeds, dim=0)
        text_ids = torch.cat(text_ids, dim=0)
        text_atts = torch.cat(text_atts, dim=0)
        text_feats = torch.cat(text_feats, dim=0)

        video_feats = []
        video_embeds = []
        for samples in data_loader:
            video = samples["video"]

            video = video.to(self.device)
            video_feat = self.visual_encoder.forward_features(video)
            video_embed = self.vision_proj(video_feat[:, 0, :])
            video_embed = F.normalize(video_embed, dim=-1)

            video_feats.append(video_feat.cpu())
            video_embeds.append(video_embed)

        video_feats = torch.cat(video_feats, dim=0)
        video_embeds = torch.cat(video_embeds, dim=0)

        sims_matrix = video_embeds @ text_embeds.t()
        score_matrix_v2t = torch.full(
            (len(data_loader.dataset.image), len(texts)), -100.0
        ).to(self.device)

        num_tasks = dist_utils.get_world_size()
        rank = dist_utils.get_rank()
        step = sims_matrix.size(0) // num_tasks + 1
        start = rank * step
        end = min(sims_matrix.size(0), start + step)

        # video-to-text
        for i, sims in enumerate(
            metric_logger.log_every(sims_matrix[start:end], 50, header)
        ):
            topk_sim, topk_idx = sims.topk(k=k_test, dim=0)

            video_feats_repeat = (
                video_feats[start + i].repeat(k_test, 1, 1).to(self.device)
            )
            video_atts_repeat = torch.ones(
                video_feats_repeat.size()[:-1], dtype=torch.long
            ).to(self.device)

            attention_mask = torch.cat([text_atts[topk_idx], video_atts_repeat], dim=1)
            embedding_output = torch.cat(
                [text_feats[topk_idx].to(self.device), video_feats_repeat], dim=1
            )

            output = self.text_encoder(
                encoder_embeds=embedding_output,
                attention_mask=attention_mask,
                return_dict=True,
                mode="fusion",
            )

            score = self.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
            score_matrix_v2t[start + i, topk_idx] = score + topk_sim

        # text-to-video
        sims_matrix = sims_matrix.t()
        score_matrix_t2v = torch.full(
            (len(texts), len(data_loader.dataset.image)), -100.0
        ).to(self.device)

        step = sims_matrix.size(0) // num_tasks + 1
        start = rank * step
        end = min(sims_matrix.size(0), start + step)

        for i, sims in enumerate(
            metric_logger.log_every(sims_matrix[start:end], 50, header)
        ):

            topk_sim, topk_idx = sims.topk(k=k_test, dim=0)

            text_feats_repeat = (
                text_feats[start + i].repeat(k_test, 1, 1).to(self.device)
            )
            text_atts_repeat = text_atts[start + i].repeat(k_test, 1).to(self.device)

            video_atts = torch.ones(
                video_feats[topk_idx].size()[:-1], dtype=torch.long
            ).to(self.device)

            embedding_output = torch.cat(
                [text_feats_repeat, video_feats[topk_idx].to(self.device)], dim=1
            )
            attention_mask = torch.cat([text_atts_repeat, video_atts], dim=1)

            output = self.text_encoder(
                encoder_embeds=embedding_output,
                attention_mask=attention_mask,
                return_dict=True,
                mode="fusion",
            )

            score = self.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
            score_matrix_t2v[start + i, topk_idx] = score + topk_sim

        if dist_utils.is_dist_avail_and_initialized():
            dist.barrier()
            torch.distributed.all_reduce(
                score_matrix_v2t, op=torch.distributed.ReduceOp.SUM
            )
            torch.distributed.all_reduce(
                score_matrix_t2v, op=torch.distributed.ReduceOp.SUM
            )

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logging.info("Evaluation time {}".format(total_time_str))

        return score_matrix_v2t.cpu().numpy(), score_matrix_t2v.cpu().numpy()

    @classmethod
    def from_config(cls, cfg):
        # vision encoder
        visual_encoder_config = node_to_dict(cfg.timesformer)
        visual_encoder = TimeSformer(**visual_encoder_config)

        # text encoder
        text_encoder = XBertEncoder.from_config(cfg)

        max_txt_len = cfg.get("max_txt_len", 35)

        model = cls(
            visual_encoder=visual_encoder,
            text_encoder=text_encoder,
            max_txt_len=max_txt_len,
        )

        num_patches = (
            visual_encoder_config["image_size"] // visual_encoder_config["patch_size"]
        ) ** 2
        num_frames = visual_encoder_config["n_frms"]

        model.load_checkpoint_from_config(
            cfg, num_frames=num_frames, num_patches=num_patches
        )

        return model
