"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import torch
import torch.nn as nn
from itertools import chain
from lavis.common.registry import registry
from lavis.models.base_model import BaseModel
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import T5ForConditionalGeneration
from lavis.models.pnp_vqa_models import prepare_qa_input
from lavis.models.blip_models.blip_image_text_matching import compute_gradcam
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions


@registry.register_model("pnp_vqa")
class PNPVQA(BaseModel):
    """
    PNPVQA model consists of three submodels for zero-shot VQA:
        1. Image-questioning matching model
        2. Image captioning model
        3. Question answering model

    Supported model types:
        - base: BLIPITM, BLIPCaption, PNPUnifiedQAv2FiD (t5-base)
        - large: BLIPITM, BLIPCaption, PNPUnifiedQAv2FiD (t5-large)
        - 3b: BLIPITM, BLIPCaption, PNPUnifiedQAv2FiD (t5-3b)

    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("pnp_vqa", "base", is_eval=True)
        >>> model = load_model("pnp_vqa", "large", is_eval=True)
        >>> model = load_model("pnp_vqa", "3b", is_eval=True)
    """

    PRETRAINED_MODEL_CONFIG_DICT = {"base": "configs/models/pnp-vqa/pnp_vqa_base.yaml",
                                    "large": "configs/models/pnp-vqa/pnp_vqa_large.yaml",
                                    "3b": "configs/models/pnp-vqa/pnp_vqa_3b.yaml",
                                    }

    def __init__(self, image_question_matching_model, image_captioning_model,
                 question_answering_model, offload_model=False):
        super().__init__()

        self.image_question_matching_model = image_question_matching_model
        self.image_captioning_model = image_captioning_model
        self.question_answering_model = question_answering_model
        self.offload_model = offload_model

    def forward_itm(self, samples, block_num=7):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
                - text_input (list): A list of strings of length batch_size
            block_num (int): The index of cross-attention block for gradcam computation.

        Returns:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
                - text_input (list): A list of strings of length batch_size
                - gradcams (torch.Tensor): A tensor of shape (batch_size, H*W)
        """
        image = samples['image']
        question = [text.strip('?') for text in samples['text_input']]
        tokenized_text = self.image_question_matching_model.tokenizer(question, padding='longest', truncation=True,
                                                return_tensors="pt").to(self.image_question_matching_model.device)
        with torch.set_grad_enabled(True):
            gradcams, _ = compute_gradcam(model=self.image_question_matching_model,
                            visual_input=image,
                            text_input=question,
                            tokenized_text=tokenized_text,
                            block_num=block_num)

        gradcams = [gradcam_[1] for gradcam_ in gradcams]
        samples['gradcams'] = torch.stack(gradcams).reshape(samples['image'].size(0), -1)

        return samples

    def forward_cap(
            self,
            samples,
            cap_max_length=20,
            cap_min_length=0,
            top_p=1,
            top_k=50,
            repetition_penalty=1.0,
            num_captions=100,
            num_patches=20,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
                - text_input (list): A list of strings of length batch_size
                - gradcams (torch.Tensor): A tensor of shape (batch_size, H*W)
            cap_max_length (int): The maximum length of the caption to be generated.
            cap_min_length (int): The minimum length of the caption to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            top_k (float): The number of the highest probability tokens for top-k sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions generated for each image.
            num_patches (int): Number of patches sampled for each image.

        Returns:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
                - text_input (list): A list of strings of length batch_size
                - gradcams (torch.Tensor): A tensor of shape (batch_size, H*W)
                - captions (nested list): A nested list of strings of total length batch_size * num_captions
        """
        encoder_out = self.image_captioning_model.forward_encoder(samples)
        captions = [[] for _ in range(encoder_out.size(0))]

        min_num_captions = 0

        while min_num_captions < num_captions:
            encoder_out_samples = []
            for i in range(num_captions):
                patch_id = torch.multinomial(samples['gradcams'].to(self.image_captioning_model.device),
                                             num_patches).reshape(encoder_out.size(0), -1) + 1
                patch_id = patch_id.sort(dim=1).values.unsqueeze(-1).expand(-1, -1, encoder_out.size(2))
                encoder_out_sample = torch.gather(encoder_out, 1, patch_id)
                encoder_out_samples.append(encoder_out_sample)

            stacked = torch.stack(encoder_out_samples, dim=1)
            image_embeds = torch.flatten(stacked, start_dim=0, end_dim=1) #(bsz*num_seq, num_patch, dim)

            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(self.image_captioning_model.device)
            model_kwargs = {
                "encoder_hidden_states": image_embeds,
                "encoder_attention_mask": image_atts,
            }

            prompt = [self.image_captioning_model.prompt] * image_embeds.size(0)
            prompt = self.image_captioning_model.tokenizer(prompt,
                                                           return_tensors="pt").to(self.image_captioning_model.device)
            prompt.input_ids[:, 0] = self.image_captioning_model.tokenizer.bos_token_id
            prompt.input_ids = prompt.input_ids[:, :-1]

            decoder_out = self.image_captioning_model.text_decoder.generate(
                input_ids=prompt.input_ids,
                max_length=cap_max_length,
                min_length=cap_min_length,
                do_sample=True,
                top_p=top_p,
                top_k=top_k,
                num_return_sequences=1,
                eos_token_id=self.image_captioning_model.tokenizer.sep_token_id,
                pad_token_id=self.image_captioning_model.tokenizer.pad_token_id,
                repetition_penalty=repetition_penalty,
                **model_kwargs)

            outputs = self.image_captioning_model.tokenizer.batch_decode(decoder_out, skip_special_tokens=True)

            for counter, output in enumerate(outputs):
                ind = counter//num_captions
                if len(captions[ind]) < num_captions:
                    caption = output[len(self.image_captioning_model.prompt):]
                    overlap_caption = [1 for caps in captions[ind] if caption in caps]
                    if len(overlap_caption) == 0:
                        captions[ind].append(caption)

            min_num_captions = min([len(i) for i in captions])

        samples['captions'] = captions

        return samples

    def forward_qa(
            self,
            samples,
            num_beams=1,
            max_len=20,
            min_len=0,
            internal_bsz_fid=1,
            num_captions=100,
            num_captions_fid=1,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
                - text_input (list): A list of strings of length batch_size
                - gradcams (torch.Tensor): A tensor of shape (batch_size, H*W)
                - captions (nested list): A nested list of strings of total length batch_size * num_captions
                - question_captions (nested list): A nested list of concatenated strings of questions and captions
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_len (int): Maximum length of generated answers.
            min_len (int): Minimum length of generated answers.
            internal_bsz_fid (int): Internal batch size when using FiD decoding.
            num_captions (int): Number of captions generated for each image.
            num_captions_fid (int): Number of captions concatenated with a question during FiD decoding.

        Returns:
            List: A list of strings, each string is an answer.
        """
        prepare_qa_input(samples, num_captions=num_captions, num_captions_fid=num_captions_fid)

        pred_answers = []
        question_captions = samples['question_captions']
        question_captions_chunk = [question_captions[i:i + internal_bsz_fid]
                                   for i in range(0, len(question_captions), internal_bsz_fid)]
        question_captions_chunk = list(chain(*question_captions_chunk))

        for question_caption in question_captions_chunk:
            question_caption_input = self.question_answering_model.tokenizer(question_caption, padding='longest',
                                        truncation=True, return_tensors="pt").to(self.question_answering_model.device)

            question_caption_input.input_ids = question_caption_input.input_ids.reshape(
                                               internal_bsz_fid, -1, question_caption_input.input_ids.size(1))
            question_caption_input.attention_mask = question_caption_input.attention_mask.reshape(
                                               internal_bsz_fid, -1, question_caption_input.attention_mask.size(1))

            outputs = self.question_answering_model.generate(input_ids=question_caption_input.input_ids,
                                            attention_mask=question_caption_input.attention_mask,
                                            num_beams=num_beams,
                                            min_length=min_len,
                                            max_length=max_len,
                                            )

            for output in outputs:
                pred_answer = self.question_answering_model.tokenizer.decode(output, skip_special_tokens=True)
                pred_answers.append(pred_answer)

        return pred_answers

    def predict_answers(
        self,
        samples,
        num_beams=1,
        inference_method="generate",
        max_len=20,
        min_len=0,
        internal_bsz_fid=1,
        num_captions=50,
        num_captions_fid=1,
        cap_max_length=20,
        cap_min_length=10,
        top_k=50,
        top_p=1,
        repetition_penalty=1,
        num_patches=50,
        block_num=7,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W). Default H=480, W=480.
                - text_input (str or [str]): String or a list of strings, each string is a question.
                                             The number of questions must be equal to the batch size. If a single string, will be converted to a list of string, with length 1 first.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            inference_method (str): Inference method. Must be "generate". The model will generate answers.
            max_len (int): Maximum length of generated answers.
            min_len (int): Minimum length of generated answers.
            internal_bsz_fid (int): Internal batch size when using FiD decoding.
            num_captions (int): Number of captions generated for each image.
            num_captions_fid (int): Number of captions concatenated with a question during FiD decoding.
            cap_max_length (int): The maximum length of the caption to be generated.
            cap_min_length (int): The minimum length of the caption to be generated.
            top_k (float): The number of the highest probability tokens for top-k sampling.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_patches (int): Number of patches sampled for each image.
            block_num (int): The index of cross-attention block for gradcam computation.

        Returns:
            List: A list of strings, each string is an answer.
            gradcams (torch.Tensor): A tensor of shape (batch_size, H*W)
            captions (nested list): A nested list of strings of total length batch_size * num_captions
        """
        assert inference_method in [
            "generate",
        ], "Inference method must be 'generate', got {}.".format(
            inference_method
        )

        if isinstance(samples["text_input"], str):
            samples["text_input"] = [samples["text_input"]]

        assert len(samples["text_input"]) == samples["image"].size(
            0
        ), "The number of questions must be equal to the batch size."

        samples = self.forward_itm(samples, block_num=block_num)

        samples = self.forward_cap(samples,
                                   cap_max_length=cap_max_length,
                                   cap_min_length=cap_min_length,
                                   top_k=top_k,
                                   top_p=top_p,
                                   repetition_penalty=repetition_penalty,
                                   num_captions=num_captions,
                                   num_patches=num_patches)

        if self.offload_model:
            samples['image'] = samples['image'].to('cpu')
            self.image_question_matching_model.to('cpu')
            self.image_captioning_model.to('cpu')
        torch.cuda.empty_cache()

        pred_answers = self.forward_qa(samples,
                                  num_beams=num_beams,
                                  max_len=max_len,
                                  min_len=min_len,
                                  internal_bsz_fid=internal_bsz_fid,
                                  num_captions=num_captions,
                                  num_captions_fid=num_captions_fid)

        if self.offload_model:
            self.image_question_matching_model.to(self.question_answering_model.device)
            self.image_captioning_model.to(self.question_answering_model.device)

        return pred_answers, samples['captions'], samples['gradcams']

    @classmethod
    def from_config(cls, model_config):
        itm_config = model_config.image_question_matching_model
        cap_config = model_config.image_captioning_model
        qa_config = model_config.question_answering_model

        itm_cls = registry.get_model_class(itm_config.arch)
        cap_cls = registry.get_model_class(cap_config.arch)
        qa_cls = registry.get_model_class(qa_config.arch)

        image_question_matching_model = itm_cls.from_config(itm_config)
        image_captioning_model = cap_cls.from_config(cap_config)
        question_answering_model = qa_cls.from_config(qa_config)

        model = cls(image_question_matching_model=image_question_matching_model,
                    image_captioning_model=image_captioning_model,
                    question_answering_model=question_answering_model,
                    offload_model= True if model_config.model_type == '3b' else False,
                    )

        return model