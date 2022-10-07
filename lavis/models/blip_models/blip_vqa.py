"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import torch
import torch.nn.functional as F
from lavis.common.registry import registry
from lavis.models.base_model import tile
from lavis.models.blip_models.blip import BlipBase
from lavis.models.blip_models.blip_outputs import (
    BlipOutput,
    BlipIntermediateOutput,
)
from lavis.models.med import XBertEncoder, XBertLMHeadDecoder
from lavis.models.vit import VisionTransformerEncoder


@registry.register_model("blip_vqa")
class BlipVQA(BlipBase):
    """
    BLIP VQA models.

    Supported model types:
        - base: vqa model initialized with pre-trained BLIP base model on 115M image-text pairs after CapFilt; not fine-tuned.
        - vqav2: fine-tuned BLIP base model on VQA v2.0 dataset.

    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip_vqa", "vqav2")
        >>> model = load_model("blip_vqa", "okvqa")
        >>> model = load_model("blip_vqa", "aokvqa")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "vqav2": "configs/models/blip_vqav2.yaml",
        "okvqa": "configs/models/blip_vqa_okvqa.yaml",
        "aokvqa": "configs/models/blip_vqa_aokvqa.yaml",
    }

    def __init__(self, image_encoder, text_encoder, text_decoder, max_txt_len=35):
        super().__init__()
        self.tokenizer = self.init_tokenizer()

        self.visual_encoder = image_encoder

        self.text_encoder = text_encoder
        self.text_decoder = text_decoder

        self.max_txt_len = max_txt_len

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
            A BlipOutput object containing loss and intermediate outputs,
            see :class:`lavis.models.blip_outputs.BlipOutput` for more details.

        Examples:
        ```python
            >>> import torch
            >>> from lavis.models import load_model
            >>> model = load_model("blip_vqa")
            >>> samples = {
            ...     "image": torch.rand(2, 3, 480, 480),
            ...     "text_input": ["What is this?", "What is that?"],
            ...     "answer": ["cat", "cat", "dog"],
            ...     "weight": torch.tensor([1.0, 1.0, 1.0]),
            ...     "n_answers": torch.tensor([2, 1]),
            ... }
            >>> output = model(samples)
            >>> output.keys()
            odict_keys(['intermediate_output', 'loss'])
            >>> output.intermediate_output.keys()
            odict_keys(['image_embeds', 'encoder_output', 'decoder_output', 'decoder_labels'])
        ```
        """
        encoder_output, image_embeds = self.forward_encoder(samples)
        loss, decoder_output, decoder_targets = self.forward_decoder(
            samples=samples, encoder_out=encoder_output
        )

        return BlipOutput(
            loss=loss,
            intermediate_output=BlipIntermediateOutput(
                image_embeds=image_embeds,
                encoder_output=encoder_output,
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
        questions.input_ids[:, 0] = self.tokenizer.enc_token_id
        samples.update({"tokenized_text": questions})

        image_embeds = self.visual_encoder.forward_features(samples["image"])
        encoder_output = self.text_encoder.forward_automask(
            tokenized_text=samples["tokenized_text"], visual_embeds=image_embeds
        )

        return encoder_output, image_embeds

    def forward_decoder(self, samples, encoder_out, **kwargs):
        answers = self.tokenizer(
            samples["answer"], padding="longest", return_tensors="pt"
        ).to(self.device)
        answers.input_ids[:, 0] = self.tokenizer.bos_token_id
        answer_targets = answers.input_ids.masked_fill(
            answers.input_ids == self.tokenizer.pad_token_id, -100
        )

        question_states = []
        question_atts = []

        question = samples["tokenized_text"]
        question_output = encoder_out

        for b, n in enumerate(samples["n_answers"]):
            question_states += [question_output.last_hidden_state[b]] * n
            question_atts += [question.attention_mask[b]] * n

        question_states = torch.stack(question_states, dim=0)
        question_atts = torch.stack(question_atts, dim=0)

        answer_output = self.text_decoder(
            answers.input_ids,
            attention_mask=answers.attention_mask,
            encoder_hidden_states=question_states,
            encoder_attention_mask=question_atts,
            labels=answer_targets,
            return_dict=True,
            reduction="none",
        )

        loss = samples["weight"] * answer_output.loss
        bsz = samples["image"].size(0)

        loss = loss.sum() / bsz

        return loss, answer_output, answer_targets

    def predict_answers(
        self,
        samples,
        num_beams=3,
        inference_method="rank",
        max_len=10,
        min_len=1,
        num_ans_candidates=128,
        answer_list=None,
        **kwargs
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W). Default H=480, W=480.
                - text_input (str or [str]): String or a list of strings, each string is a question.
                                             The number of questions must be equal to the batch size. If a single string, will be converted to a list of string, with length 1 first.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            inference_method (str): Inference method. One of "rank", "generate".
                - If "rank", the model will return answers with the highest probability from the answer list.
                - If "generate", the model will generate answers.
            max_len (int): Maximum length of generated answers.
            min_len (int): Minimum length of generated answers.
            num_ans_candidates (int): Number of answer candidates, used to filter out answers with low probability.
            answer_list (list): A list of strings, each string is an answer.

        Returns:
            List: A list of strings, each string is an answer.

        Examples:
        ```python
            >>> from PIL import Image
            >>> from lavis.models import load_model_and_preprocess
            >>> model, vis_processors, txt_processors = load_model_and_preprocess("blip_vqa", "vqav2")
            >>> raw_image = Image.open("docs/data/merlion.png").convert("RGB")
            >>> question = "Which city is this photo taken?"
            >>> image = vis_processors["eval"](raw_image).unsqueeze(0)
            >>> question = txt_processors["eval"](question)
            >>> samples = {"image": image, "text_input": [question]}
            >>> answers = model.predict_answers(samples)
            >>> answers
            ['singapore']
            >>> answer_list = ["Singapore", "London", "Palo Alto", "Tokyo"]
            >>> answers = model.predict_answers(samples, answer_list=answer_list)
            >>> answers
            ['Singapore']
        ```
        """
        assert inference_method in [
            "rank",
            "generate",
        ], "Inference method must be one of 'rank' or 'generate', got {}.".format(
            inference_method
        )

        if isinstance(samples["text_input"], str):
            samples["text_input"] = [samples["text_input"]]

        assert len(samples["text_input"]) == samples["image"].size(
            0
        ), "The number of questions must be equal to the batch size."

        if inference_method == "generate":
            return self._generate_answers(
                samples, num_beams=num_beams, max_length=max_len, min_length=min_len
            )
        elif inference_method == "rank":
            assert answer_list is not None, "answer_list must be provided for ranking"

            num_ans_candidates = min(num_ans_candidates, len(answer_list))

            return self._rank_answers(
                samples, answer_list=answer_list, num_ans_candidates=num_ans_candidates
            )

    def _generate_answers(self, samples, num_beams=3, max_length=10, min_length=1):
        encoder_out, _ = self.forward_encoder(samples)

        question_output = encoder_out

        question_states = question_output.last_hidden_state.repeat_interleave(
            num_beams, dim=0
        )
        question_atts = torch.ones(question_states.size()[:-1], dtype=torch.long).to(
            self.device
        )

        model_kwargs = {
            "encoder_hidden_states": question_states,
            "encoder_attention_mask": question_atts,
        }

        bsz = samples["image"].size(0)
        bos_ids = torch.full(
            (bsz, 1), fill_value=self.tokenizer.bos_token_id, device=self.device
        )

        outputs = self.text_decoder.generate(
            input_ids=bos_ids,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            eos_token_id=self.tokenizer.sep_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            **model_kwargs
        )

        # collect answers
        answers = []
        for output in outputs:
            answer = self.tokenizer.decode(output, skip_special_tokens=True)
            answers.append(answer)

        return answers

    def _rank_answers(self, samples, answer_list, num_ans_candidates):
        """
        Generate the first token of answers using decoder and select ${num_ans_candidates}
        most probable ones. Then select answers from answer list, which start with the probable tokens.
        Lastly, use the selected answers as the ground-truth labels for decoding and calculating LM loss.
        Return the answers that minimize the losses as result.

        """
        answer_candidates = self.tokenizer(
            answer_list, padding="longest", return_tensors="pt"
        ).to(self.device)
        answer_candidates.input_ids[:, 0] = self.tokenizer.bos_token_id

        answer_ids = answer_candidates.input_ids
        answer_atts = answer_candidates.attention_mask

        question_output, _ = self.forward_encoder(samples)
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

        # text encoder + multimodal encoder
        text_encoder = XBertEncoder.from_config(cfg)
        text_decoder = XBertLMHeadDecoder.from_config(cfg)

        max_txt_len = cfg.get("max_txt_len", 35)

        model = cls(
            image_encoder=image_encoder,
            text_encoder=text_encoder,
            text_decoder=text_decoder,
            max_txt_len=max_txt_len,
        )

        model.load_checkpoint_from_config(cfg)

        return model
