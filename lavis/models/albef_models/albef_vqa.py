"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
import os
from copy import deepcopy

import torch
import torch.nn.functional as F
from lavis.common.registry import registry
from lavis.common.utils import get_abs_path, is_url
from lavis.models.albef_models import AlbefBase
from lavis.models.albef_models.albef_outputs import AlbefIntermediateOutput, AlbefOutput
from lavis.models.base_model import MomentumDistilationMixin, tile
from lavis.models.med import BertConfig, BertLMHeadModel, XBertEncoder
from lavis.models.vit import VisionTransformerEncoder, interpolate_pos_embed
from lavis.common.dist_utils import download_cached_file


@registry.register_model("albef_vqa")
class AlbefVQA(AlbefBase, MomentumDistilationMixin):
    """
    ALBEF VQA models.

    Supported model types:
        - base: vqa model initialized with pre-trained ALBEF base model on 115M image-text pairs after CapFilt; not fine-tuned.
        - vqav2: fine-tuned ALBEF base model on VQA v2.0 dataset.

    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("albef_vqa", "vqav2")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "vqav2": "configs/models/albef_vqav2.yaml",
    }

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

        self.tokenizer = self.init_tokenizer()
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

    def _rampup_factor(self, epoch, iters, num_iters_per_epoch):
        return min(1, (epoch * num_iters_per_epoch + iters) / num_iters_per_epoch)

    def forward(self, samples):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W). Default H=480, W=480.
                - text_input (list): A list of strings, each string is a question
                - answer (list): A list of strings, each string is an answer
                - weight (torch.Tensor): A tensor used to weigh each answer in the loss computation.
                   The shape of the tensor is (sum(n_answers),)
                - n_answers (torch.Tensor): A tensor shape (batch_size,) containing the number of answers
                     for each question in the batch.

        Returns:
            An AlbefOutput object containing loss and intermediate outputs;
            see lavis/models/albef_models/albef_outputs.py for more details.

        Examples:
            >>> import torch
            >>> from lavis.models import load_model
            >>> model = load_model("albef_vqa")
            >>> samples = {
            ...     "image": torch.rand(2, 3, 384, 384),
            ...     "text_input": ["What is this?", "What is that?"],
            ...     "answer": ["cat", "cat", "dog"],
            ...     "weight": torch.tensor([1.0, 1.0, 1.0]),
            ...     "n_answers": torch.tensor([2, 1]),
            ...     "epoch": 0, "iters": 0, "num_iters_per_epoch": 1000,
            ... }
            >>> output = model(samples)
            >>> output.keys()
            odict_keys(['intermediate_output', 'loss'])
        """
        (
            encoder_output,
            encoder_output_m,
            image_embeds,
            image_embeds_m,
        ) = self.forward_encoder(samples)
        loss, decoder_output, decoder_targets = self.forward_decoder(
            samples, encoder_out=(encoder_output, encoder_output_m)
        )

        return AlbefOutput(
            loss=loss,
            intermediate_output=AlbefIntermediateOutput(
                image_embeds=image_embeds,
                image_embeds_m=image_embeds_m,
                encoder_output=encoder_output,
                encoder_output_m=encoder_output_m,
                decoder_output=decoder_output,
                decoder_labels=decoder_targets,
            ),
        )

    def forward_encoder(self, samples):
        questions = samples["text_input"]
        questions = self.tokenizer(
            questions,
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(self.device)
        samples.update({"tokenized_text": questions})

        image_embeds = self.visual_encoder.forward_features(samples["image"])
        encoder_output = self.text_encoder.forward_automask(
            tokenized_text=samples["tokenized_text"], visual_embeds=image_embeds
        )

        if self.use_distill:
            self._momentum_update()
            with torch.no_grad():
                image_embeds_m = self.visual_encoder_m(samples["image"])
                encoder_output_m = self.text_encoder_m.forward_automask(
                    tokenized_text=samples["tokenized_text"],
                    visual_embeds=image_embeds_m,
                )
        else:
            encoder_output_m = None
            image_embeds_m = None

        return encoder_output, encoder_output_m, image_embeds, image_embeds_m

    def forward_decoder(self, samples, encoder_out, **kwargs):
        answers = self.tokenizer(
            samples["answer"], padding="longest", return_tensors="pt"
        ).to(self.device)
        answer_targets = answers.input_ids.masked_fill(
            answers.input_ids == self.tokenizer.pad_token_id, -100
        )

        question_states = []
        question_atts = []

        question = samples["tokenized_text"]
        question_output, question_output_m = encoder_out

        for b, n in enumerate(samples["n_answers"]):
            question_states += [question_output.last_hidden_state[b]] * n
            question_atts += [question.attention_mask[b]] * n

        question_states = torch.stack(question_states, dim=0)
        question_atts = torch.stack(question_atts, dim=0)

        if self.use_distill:
            with torch.no_grad():
                question_states_m = []
                for b, n in enumerate(samples["n_answers"]):
                    question_states_m += [question_output_m.last_hidden_state[b]] * n
                question_states_m = torch.stack(question_states_m, 0)

                logits_m = self.text_decoder_m(
                    answers.input_ids,
                    attention_mask=answers.attention_mask,
                    encoder_hidden_states=question_states_m,
                    encoder_attention_mask=question_atts,
                    return_logits=True,
                )

                alpha = self.alpha * self._rampup_factor(
                    epoch=samples["epoch"],
                    iters=samples["iters"],
                    num_iters_per_epoch=samples["num_iters_per_epoch"],
                )

        answer_output = self.text_decoder(
            answers.input_ids,
            attention_mask=answers.attention_mask,
            encoder_hidden_states=question_states,
            encoder_attention_mask=question_atts,
            labels=answer_targets,
            soft_labels=F.softmax(logits_m, dim=-1),
            alpha=alpha,
            return_dict=True,
            reduction="none",
        )

        loss = samples["weight"] * answer_output.loss
        bsz = samples["image"].size(0)

        loss = loss.sum() / bsz

        return loss, answer_output, answer_targets

    def predict_answers(self, samples, answer_list, num_ans_candidates=128, **kwargs):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W). Default H=480, W=480.
                - text_input (str or [str]): String or a list of strings, each string is a question.
                                             The number of questions must be equal to the batch size. If a single string, will be converted to a list of string, with length 1 first.
            num_ans_candidates (int): Number of answer candidates, used to filter out answers with low probability.
            answer_list (list): A list of strings, each string is an answer.

        Returns:
            List: A list of strings, each string is an answer.

        Examples:
            >>> from PIL import Image
            >>> from lavis.models import load_model_and_preprocess
            >>> model, vis_processors, txt_processors = load_model_and_preprocess("albef_vqa", "vqav2")
            >>> raw_image = Image.open("docs/data/merlion.png").convert("RGB")
            >>> question = "Which city is this photo taken?"
            >>> image = vis_processors["eval"](raw_image).unsqueeze(0)
            >>> question = txt_processors["eval"](question)
            >>> samples = {"image": image, "text_input": [question]}
            >>> answer_list = ["Singapore", "London", "Palo Alto", "Tokyo"]
            >>> answers = model.predict_answers(samples, answer_list=answer_list)
            >>> answers
            ['Singapore']
        """

        if isinstance(samples["text_input"], str):
            samples["text_input"] = [samples["text_input"]]

        assert len(samples["text_input"]) == samples["image"].size(
            0
        ), "The number of questions must be equal to the batch size."

        num_ans_candidates = min(num_ans_candidates, len(answer_list))

        return self.rank_answers(
            samples, answer_list=answer_list, num_ans_candidates=num_ans_candidates
        )

    def rank_answers(self, samples, answer_list, num_ans_candidates):
        """
        Generate the first token of answers using decoder and select ${num_ans_candidates}
        most probable ones. Then select answers from answer list, which start with the probable tokens.
        Lastly, use the selected answers as the ground-truth labels for decoding and calculating LM loss.
        Return the answers that minimize the losses as result.

        """
        answer_candidates = self.tokenizer(
            answer_list, padding="longest", return_tensors="pt"
        ).to(self.device)
        # answer_candidates.input_ids[:, 0] = self.tokenizer.bos_token_id

        answer_ids = answer_candidates.input_ids
        answer_atts = answer_candidates.attention_mask

        question_output, _, _, _ = self.forward_encoder(samples)
        question_states = question_output.last_hidden_state

        tokenized_question = samples["tokenized_text"]
        question_atts = tokenized_question.attention_mask

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
        topk_probs, topk_ids = prob_first_token.topk(num_ans_candidates, dim=1)

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
        question_states = tile(question_states, 0, num_ans_candidates)
        question_atts = tile(question_atts, 0, num_ans_candidates)

        output = self.text_decoder(
            input_ids,
            attention_mask=input_atts,
            encoder_hidden_states=question_states,
            encoder_attention_mask=question_atts,
            labels=targets_ids,
            return_dict=True,
            reduction="none",
        )

        log_probs_sum = -output.loss
        log_probs_sum = log_probs_sum.view(num_ques, num_ans_candidates)

        max_topk_ids = log_probs_sum.argmax(dim=1)
        max_ids = topk_ids[max_topk_ids >= 0, max_topk_ids]

        answers = [answer_list[max_id] for max_id in max_ids]

        return answers

    @classmethod
    def from_config(cls, cfg=None):
        image_encoder = VisionTransformerEncoder.from_config(cfg)

        text_encoder = XBertEncoder.from_config(cfg)

        config_decoder = BertConfig.from_json_file(get_abs_path(cfg["med_config_path"]))
        config_decoder.fusion_layer = 0
        config_decoder.num_hidden_layers = 6
        text_decoder = BertLMHeadModel.from_pretrained(
            "bert-base-uncased", config=config_decoder
        )

        alpha = cfg.get("alpha", 0.4)
        momentum = cfg.get("momentum", 0.995)
        use_distill = cfg.get("use_distill", True)
        max_txt_len = cfg.get("max_txt_len", 25)

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

        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        # reshape positional embedding to accomodate for image resolution change
        pos_embed_reshaped = interpolate_pos_embed(
            state_dict["visual_encoder.pos_embed"], self.visual_encoder
        )
        state_dict["visual_encoder.pos_embed"] = pos_embed_reshaped

        m_pos_embed_reshaped = interpolate_pos_embed(
            state_dict["visual_encoder_m.pos_embed"], self.visual_encoder_m
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

        for key in self.state_dict().keys():
            if key in state_dict.keys():
                if state_dict[key].shape != self.state_dict()[key].shape:
                    del state_dict[key]

        msg = self.load_state_dict(state_dict, strict=False)
        logging.info("load checkpoint from %s" % url_or_filename)
        logging.info(f"missing keys: {msg.missing_keys}")

        return msg
