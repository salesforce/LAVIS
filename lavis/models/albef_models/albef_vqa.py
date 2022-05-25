from copy import deepcopy
import logging
import os

import torch
import torch.nn.functional as F
from common.registry import registry
from common.utils import is_url
from models.albef_models import (
    MomentumDistilationMixin,
    init_tokenizer,
)
from models.base_model import BaseModel
from models.blip_models import tile
from models.med import BertConfig, BertLMHeadModel, XBertEncoder
from models.vit import VisionTransformerEncoder, interpolate_pos_embed
from timm.models.hub import download_cached_file


@registry.register_model("albef_vqa")
class AlbefVQA(BaseModel, MomentumDistilationMixin):
    def __init__(
        self,
        image_encoder,
        text_encoder,
        text_decoder,
        use_distill=True,
        momentum=0.995,
        alpha=0.4,
        max_txt_len=35,
    ):
        super().__init__()

        self.tokenizer = init_tokenizer()
        self.max_txt_len = max_txt_len

        self.use_distill = use_distill

        self.visual_encoder = image_encoder

        self.text_encoder = text_encoder
        self.text_decoder = text_decoder

        if self.use_distill:
            self.visual_encoder_m = deepcopy(self.visual_encoder)
            self.text_encoder_m = deepcopy(self.text_encoder)
            self.text_decoder_m = deepcopy(self.text_decoder)

            self.momentum = momentum
            self.alpha = alpha

            self.model_pairs = [
                [self.visual_encoder, self.visual_encoder_m],
                [self.text_encoder, self.text_encoder_m],
                [self.text_decoder, self.text_decoder_m],
            ]

            self.copy_params()

    @classmethod
    def default_config_path(cls, model_type="base"):
        paths = {
            "base": "configs/models/albef_vqa_base.yaml",
            # "large": "configs/models/blip_pretrain_large.yaml"
        }

        assert model_type in paths, "Unknown model type {}".format(model_type)
        return paths[model_type]

    def forward(
        self, image, quesiton, answer=None, alpha=0, k=None, weights=None, train=True
    ):

        image_embeds = self.visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        if train:
            """
            k: number of answers for each question
            weights: weight for each answer
            """
            answer_targets = answer.input_ids.masked_fill(
                answer.input_ids == self.tokenizer.pad_token_id, -100
            )

            question_output = self.text_encoder(
                quesiton.input_ids,
                attention_mask=quesiton.attention_mask,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            question_states = []
            question_atts = []
            for b, n in enumerate(k):
                question_states += [question_output.last_hidden_state[b]] * n
                question_atts += [quesiton.attention_mask[b]] * n
            question_states = torch.stack(question_states, 0)
            question_atts = torch.stack(question_atts, 0)

            if self.distill:
                with torch.no_grad():
                    self._momentum_update()
                    image_embeds_m = self.visual_encoder_m(image)
                    question_output_m = self.text_encoder_m(
                        quesiton.input_ids,
                        attention_mask=quesiton.attention_mask,
                        encoder_hidden_states=image_embeds_m,
                        encoder_attention_mask=image_atts,
                        return_dict=True,
                    )

                    question_states_m = []
                    for b, n in enumerate(k):
                        question_states_m += [
                            question_output_m.last_hidden_state[b]
                        ] * n
                    question_states_m = torch.stack(question_states_m, 0)

                    logits_m = self.text_decoder_m(
                        answer.input_ids,
                        attention_mask=answer.attention_mask,
                        encoder_hidden_states=question_states_m,
                        encoder_attention_mask=question_atts,
                        return_logits=True,
                    )

                answer_output = self.text_decoder(
                    answer.input_ids,
                    attention_mask=answer.attention_mask,
                    encoder_hidden_states=question_states,
                    encoder_attention_mask=question_atts,
                    labels=answer_targets,
                    return_dict=True,
                    soft_labels=F.softmax(logits_m, dim=-1),
                    reduction="none",
                )
            else:
                answer_output = self.text_decoder(
                    answer.input_ids,
                    attention_mask=answer.attention_mask,
                    encoder_hidden_states=question_states,
                    encoder_attention_mask=question_atts,
                    labels=answer_targets,
                    return_dict=True,
                    reduction="none",
                )
            loss = weights * answer_output.loss
            loss = loss.sum() / image.size(0)

            return loss

        else:
            question_output = self.text_encoder(
                quesiton.input_ids,
                attention_mask=quesiton.attention_mask,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            topk_ids, topk_probs = self.rank_answer(
                question_output.last_hidden_state,
                quesiton.attention_mask,
                answer.input_ids,
                answer.attention_mask,
                k,
            )
            return topk_ids, topk_probs

    def rank_answer(self, question_states, question_atts, answer_ids, answer_atts, k):

        num_ques = question_states.size(0)
        start_ids = answer_ids[0, 0].repeat(num_ques, 1)  # bos token

        start_output = self.text_decoder(
            start_ids,
            encoder_hidden_states=question_states,
            encoder_attention_mask=question_atts,
            return_dict=True,
            reduction="none",
        )
        logits = start_output.logits[:, 0, :]  # first token's logit

        # topk_probs: top-k probability
        # topk_ids: [num_question, k]
        answer_first_token = answer_ids[:, 1]
        prob_first_token = F.softmax(logits, dim=1).index_select(
            dim=1, index=answer_first_token
        )
        topk_probs, topk_ids = prob_first_token.topk(k, dim=1)

        # answer input: [num_question*k, answer_len]
        input_ids = []
        input_atts = []
        for b, topk_id in enumerate(topk_ids):
            input_ids.append(answer_ids.index_select(dim=0, index=topk_id))
            input_atts.append(answer_atts.index_select(dim=0, index=topk_id))
        input_ids = torch.cat(input_ids, dim=0)
        input_atts = torch.cat(input_atts, dim=0)

        targets_ids = input_ids.masked_fill(
            input_ids == self.tokenizer.pad_token_id, -100
        )

        # repeat encoder's output for top-k answers
        question_states = tile(question_states, 0, k)
        question_atts = tile(question_atts, 0, k)

        output = self.text_decoder(
            input_ids,
            attention_mask=input_atts,
            encoder_hidden_states=question_states,
            encoder_attention_mask=question_atts,
            labels=targets_ids,
            return_dict=True,
            reduction="none",
        )

        answer_loss = output.loss
        answer_loss = answer_loss.view(input_ids.size(0), -1)

        # topk_prob: first token probability
        topk_probs = topk_probs.view(-1, 1)
        log_probs = torch.cat([topk_probs.log(), -answer_loss], dim=1)

        # re-calculate log probabilities for the answer sequences using chain rule
        log_probs_sum = log_probs.sum(1)
        log_probs_sum = log_probs_sum.view(num_ques, k)

        topk_probs = F.softmax(log_probs_sum, dim=-1)
        # get top-k after re-ranking
        topk_probs, rerank_id = topk_probs.topk(k, dim=1)
        topk_ids = torch.gather(topk_ids, 1, rerank_id)

        return topk_ids, topk_probs

    @classmethod
    def _build_from_cfg(cls, cfg=None):
        image_encoder = VisionTransformerEncoder.build_from_cfg(cfg)

        text_encoder = XBertEncoder.build_from_cfg(cfg)
        # text_decoder = XBertLMHeadDecoder.build_from_cfg(cfg)

        config_decoder = BertConfig.from_json_file(cfg["med_config_path"])
        config_decoder.fusion_layer = 0
        config_decoder.num_hidden_layers = 6
        text_decoder = BertLMHeadModel.from_pretrained(
            "bert-base-uncased", config=config_decoder
        )

        alpha = cfg.get("alpha", 0.4)
        momentum = cfg.get("momentum", 0.995)
        use_distill = cfg.get("use_distill", True)
        max_txt_len = cfg.get("max_txt_len", 40)

        init_decoder_as_encoder = cfg.get("init_decoder_as_encoder")

        model = cls(
            image_encoder=image_encoder,
            text_encoder=text_encoder,
            text_decoder=text_decoder,
            use_distill=use_distill,
            momentum=momentum,
            alpha=alpha,
            max_txt_len=max_txt_len,
        )

        # load pre-trained weights
        pretrain_path = cfg.get("pretrained", None)
        if pretrain_path is not None:
            model, msg = load_from_pretrained(
                model,
                url_or_filename=pretrain_path,
            )

        return model


def load_from_pretrained(model, url_or_filename):
    if is_url(url_or_filename):
        cached_file = download_cached_file(
            url_or_filename, check_hash=False, progress=True
        )
        checkpoint = torch.load(cached_file, map_location="cpu")
    elif os.path.isfile(url_or_filename):
        checkpoint = torch.load(url_or_filename, map_location="cpu")
    else:
        raise RuntimeError("checkpoint url or path is invalid")

    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    # reshape positional embedding to accomodate for image resolution change
    pos_embed_reshaped = interpolate_pos_embed(
        state_dict["visual_encoder.pos_embed"], model.visual_encoder
    )
    state_dict["visual_encoder.pos_embed"] = pos_embed_reshaped

    m_pos_embed_reshaped = interpolate_pos_embed(
        state_dict["visual_encoder_m.pos_embed"], model.visual_encoder_m
    )
    state_dict["visual_encoder_m.pos_embed"] = m_pos_embed_reshaped

    for key in list(state_dict.keys()):
        if "bert" in key:
            encoder_key = key.replace("bert.", "")
            state_dict[encoder_key] = state_dict[key]

        # intialize text decoder as multimodal encoder (last 6 layers of model.text_encoder)
        if "text_encoder" in key:
            if "layer" in key:
                encoder_keys = key.split(".")
                layer_num = int(encoder_keys[4])
                if layer_num < 6:
                    del state_dict[key]
                    continue
                else:
                    decoder_layer_num = layer_num - 6
                    encoder_keys[4] = str(decoder_layer_num)
                    encoder_key = ".".join(encoder_keys)
            else:
                encoder_key = key
            decoder_key = encoder_key.replace("text_encoder", "text_decoder")
            state_dict[decoder_key] = state_dict[key]

            del state_dict[key]

    for key in model.state_dict().keys():
        if key in state_dict.keys():
            if state_dict[key].shape != model.state_dict()[key].shape:
                del state_dict[key]

    msg = model.load_state_dict(state_dict, strict=False)
    logging.info("load checkpoint from %s" % url_or_filename)
    logging.info(msg.missing_keys)

    return model, msg


# def tile(x, dim, n_tile):
#     init_dim = x.size(dim)
#     repeat_idx = [1] * x.dim()
#     repeat_idx[dim] = n_tile
#     x = x.repeat(*(repeat_idx))
#     order_index = torch.LongTensor(
#         np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])
#     )
#     return torch.index_select(x, dim, order_index.to(x.device))
