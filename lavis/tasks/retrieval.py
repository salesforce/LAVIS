import datetime
import logging
import time

import common.utils as utils
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from common.registry import registry

from tasks.base_task import BaseTask


@registry.register_task("retrieval")
class RetrievalTask(BaseTask):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg

        return cls(cfg=run_cfg)

    def evaluation(self, model, data_loader, **kwargs):
        score_i2t, score_t2i = self.compute_sim_matrix(model, data_loader)

        if utils.is_main_process():
            eval_result = self.itm_eval(
                score_i2t,
                score_t2i,
                data_loader.dataset.txt2img,
                data_loader.dataset.img2txt,
            )
            logging.info(eval_result)
        else:
            eval_result = None

        return eval_result

    def after_evaluation(self, val_result, **kwargs):
        return val_result

    def compute_sim_matrix(self, model, data_loader):
        config = self.cfg

        metric_logger = utils.MetricLogger(delimiter="  ")
        header = "Evaluation:"

        logging.info("Computing features for evaluation...")
        start_time = time.time()

        texts = data_loader.dataset.text
        num_text = len(texts)
        text_bs = 256
        text_ids = []
        text_embeds = []
        text_atts = []
        for i in range(0, num_text, text_bs):
            text = texts[i : min(num_text, i + text_bs)]
            text_input = model.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=35,
                return_tensors="pt",
            ).to(model.device)
            # text_output = model.text_encoder.forward_text_embeds(text_input.input_ids, attention_mask = text_input.attention_mask, mode='text')
            text_output = model.text_encoder.forward_text_embeds(text_input)
            text_embed = F.normalize(
                model.text_proj(text_output.last_hidden_state[:, 0, :])
            )
            text_embeds.append(text_embed)
            text_ids.append(text_input.input_ids)
            text_atts.append(text_input.attention_mask)

        text_embeds = torch.cat(text_embeds, dim=0)
        text_ids = torch.cat(text_ids, dim=0)
        text_atts = torch.cat(text_atts, dim=0)
        if hasattr(model.tokenizer, "enc_token_id"):
            text_ids[:, 0] = model.tokenizer.enc_token_id

        image_feats = []
        image_embeds = []
        for samples in data_loader:
            image = samples["image"]

            image = image.to(model.device)
            image_feat = model.visual_encoder(image)
            image_embed = model.vision_proj(image_feat[:, 0, :])
            image_embed = F.normalize(image_embed, dim=-1)

            image_feats.append(image_feat.cpu())
            image_embeds.append(image_embed)

        image_feats = torch.cat(image_feats, dim=0)
        image_embeds = torch.cat(image_embeds, dim=0)

        sims_matrix = image_embeds @ text_embeds.t()
        score_matrix_i2t = torch.full(
            (len(data_loader.dataset.image), len(texts)), -100.0
        ).to(model.device)

        num_tasks = utils.get_world_size()
        rank = utils.get_rank()
        step = sims_matrix.size(0) // num_tasks + 1
        start = rank * step
        end = min(sims_matrix.size(0), start + step)

        for i, sims in enumerate(
            metric_logger.log_every(sims_matrix[start:end], 50, header)
        ):
            topk_sim, topk_idx = sims.topk(k=config["k_test"], dim=0)

            encoder_output = (
                image_feats[start + i].repeat(config["k_test"], 1, 1).to(model.device)
            )
            encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(
                model.device
            )
            output = model.text_encoder.forward_bert(
                text_ids[topk_idx],
                attention_mask=text_atts[topk_idx],
                encoder_hidden_states=encoder_output,
                encoder_attention_mask=encoder_att,
                return_dict=True,
            )
            score = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
            score_matrix_i2t[start + i, topk_idx] = score + topk_sim

        sims_matrix = sims_matrix.t()
        score_matrix_t2i = torch.full(
            (len(texts), len(data_loader.dataset.image)), -100.0
        ).to(model.device)

        step = sims_matrix.size(0) // num_tasks + 1
        start = rank * step
        end = min(sims_matrix.size(0), start + step)

        for i, sims in enumerate(
            metric_logger.log_every(sims_matrix[start:end], 50, header)
        ):

            topk_sim, topk_idx = sims.topk(k=config["k_test"], dim=0)
            encoder_output = image_feats[topk_idx].to(model.device)
            encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(
                model.device
            )
            output = model.text_encoder.forward_bert(
                text_ids[start + i].repeat(config["k_test"], 1),
                attention_mask=text_atts[start + i].repeat(config["k_test"], 1),
                encoder_hidden_states=encoder_output,
                encoder_attention_mask=encoder_att,
                return_dict=True,
            )
            score = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
            score_matrix_t2i[start + i, topk_idx] = score + topk_sim

        if utils.is_dist_avail_and_initialized():
            dist.barrier()
            torch.distributed.all_reduce(
                score_matrix_i2t, op=torch.distributed.ReduceOp.SUM
            )
            torch.distributed.all_reduce(
                score_matrix_t2i, op=torch.distributed.ReduceOp.SUM
            )

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logging.info("Evaluation time {}".format(total_time_str))

        return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()

    @staticmethod
    @torch.no_grad()
    def itm_eval(scores_i2t, scores_t2i, txt2img, img2txt):

        # Images->Text
        ranks = np.zeros(scores_i2t.shape[0])
        for index, score in enumerate(scores_i2t):
            inds = np.argsort(score)[::-1]
            # Score
            rank = 1e20
            for i in img2txt[index]:
                tmp = np.where(inds == i)[0][0]
                if tmp < rank:
                    rank = tmp
            ranks[index] = rank

        # Compute metrics
        tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
        tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
        tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

        # Text->Images
        ranks = np.zeros(scores_t2i.shape[0])

        for index, score in enumerate(scores_t2i):
            inds = np.argsort(score)[::-1]
            ranks[index] = np.where(inds == txt2img[index])[0][0]

        # Compute metrics
        ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
        ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
        ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

        tr_mean = (tr1 + tr5 + tr10) / 3
        ir_mean = (ir1 + ir5 + ir10) / 3
        r_mean = (tr_mean + ir_mean) / 2

        agg_metrics = (tr1 + tr5 + tr10) / 3

        eval_result = {
            "txt_r1": tr1,
            "txt_r5": tr5,
            "txt_r10": tr10,
            "txt_r_mean": tr_mean,
            "img_r1": ir1,
            "img_r5": ir5,
            "img_r10": ir10,
            "img_r_mean": ir_mean,
            "r_mean": r_mean,
            "agg_metrics": agg_metrics,
        }
        return eval_result
