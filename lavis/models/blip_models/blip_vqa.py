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
    PRETRAINED_MODEL_CONFIG_DICT = {
        "base": "configs/models/blip_vqa_base.yaml",
        "vqav2": "configs/models/blip_vqav2.yaml",
        # "large": "configs/models/blip_vqa_large.yaml"
    }

    def __init__(self, image_encoder, text_encoder, text_decoder, max_txt_len=35):
        super().__init__()
        self.tokenizer = self.init_tokenizer()

        self.visual_encoder = image_encoder

        self.text_encoder = text_encoder
        self.text_decoder = text_decoder

        self.max_txt_len = max_txt_len

    def forward(self, samples):
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
        num_ans_candidates=None,
        answer_list=None,
        **kwargs
    ):
        if inference_method == "generate":
            return self.generate_answers(
                samples, num_beams=num_beams, max_length=max_len, min_length=min_len
            )
        else:
            return self.rank_answers(
                samples, answer_list=answer_list, num_ans_candidates=num_ans_candidates
            )

    def generate_answers(self, samples, num_beams=3, max_length=10, min_length=1):
        encoder_out = self.forward_encoder(samples)

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

        # load pre-trained weights
        pretrain_path = cfg.get("pretrained", None)
        if pretrain_path is not None:
            msg = model.load_from_pretrained(url_or_filename=pretrain_path)

        return model
