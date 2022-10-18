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
from lavis.models.pnp_vqa_models import compute_gradcam, prepare_qa_input
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions


@registry.register_model("pnp_vqa")
class PNPVQA(BaseModel):
    # pretrained_model_config_dict will be utilized for pnpvqa, but not its sub-model.
    PRETRAINED_MODEL_CONFIG_DICT = {"base": "configs/models/pnp_vqa_base.yaml",
                                    "large": "configs/models/pnp_vqa_large.yaml",
                                    "3b": "configs/models/pnp_vqa_3b.yaml",
                                    }

    def __init__(self, itm_cls, cap_cls, qa_cls,
                    itm_config, cap_config, qa_config):
        super().__init__()

        self.image_question_matching = itm_cls.from_config(itm_config)
        self.image_captioning = cap_cls.from_config(cap_config)
        self.question_answering = qa_cls.from_config(qa_config)

    def forward_itm(self, samples, block_num):
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
        tokenized_text = self.image_question_matching.tokenizer(question, padding='longest', truncation=True,
                                                return_tensors="pt").to(self.device)
        with torch.set_grad_enabled(True):
            gradcams = compute_gradcam(model=self.image_question_matching,
                            visual_input=image,
                            text_input=question,
                            tokenized_text=tokenized_text,
                            block_num=block_num)

        samples['gradcams'] = torch.stack(gradcams).reshape(samples['image'].size(0), -1)

        return samples

    def forward_cap(
            self,
            samples,
            cap_max_length=20,
            cap_min_length=0,
            top_k=50,
            top_p=1,
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
            cap_max_length (int): The maximum length of the sequence to be generated.
            cap_min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            top_k (float): The number of the highest probability tokens for top-k sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
            num_patches (int): Number of patches to be sampled for each image.

        Returns:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
                - text_input (list): A list of strings of length batch_size
                - gradcams (torch.Tensor): A tensor of shape (batch_size, H*W)
                - captions (nested list): A nested list of strings of total length batch_size * num_captions
        """

        encoder_out = self.image_captioning.forward_encoder(samples)
        captions = [[] for _ in range(encoder_out.size(0))]

        min_num_captions = 0

        while min_num_captions < num_captions:
            encoder_out_samples = []
            for i in range(num_captions):
                patch_id = torch.multinomial(samples['gradcams'].to(self.device), num_patches).reshape(encoder_out.size(0), -1) + 1
                patch_id = patch_id.sort(dim=1).values.unsqueeze(-1).expand(-1, -1, encoder_out.size(2))
                encoder_out_sample = torch.gather(encoder_out, 1, patch_id)
                encoder_out_samples.append(encoder_out_sample)

            stacked = torch.stack(encoder_out_samples, dim=1)
            image_embeds = torch.flatten(stacked, start_dim=0, end_dim=1) #(bsz*num_seq, num_patch, dim)

            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(self.device)
            model_kwargs = {
                "encoder_hidden_states": image_embeds,
                "encoder_attention_mask": image_atts,
            }

            # image_embeds = torch.repeat_interleave(encoder_out, num_captions, 0)

            ### reason not using num_return_sequence from huggingface? maybe there are the same.
            prompt = [self.image_captioning.prompt] * image_embeds.size(0)
            prompt = self.image_captioning.tokenizer(prompt, return_tensors="pt").to(self.device)
            prompt.input_ids[:, 0] = self.image_captioning.tokenizer.bos_token_id
            prompt.input_ids = prompt.input_ids[:, :-1]

            decoder_out = self.image_captioning.text_decoder.generate(
                input_ids=prompt.input_ids,
                max_length=cap_max_length,
                min_length=cap_min_length,
                do_sample=True,
                top_p=top_p,
                top_k=top_k,
                num_return_sequences=1,
                eos_token_id=self.image_captioning.tokenizer.sep_token_id,
                pad_token_id=self.image_captioning.tokenizer.pad_token_id,
                repetition_penalty=repetition_penalty,
                **model_kwargs)

            outputs = self.image_captioning.tokenizer.batch_decode(decoder_out, skip_special_tokens=True)
            for counter, output in enumerate(outputs):
                ind = counter//num_captions
                if len(captions[ind]) < num_captions:
                    caption = output[len(self.image_captioning.prompt):]
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

        Returns:
            List: A list of strings, each string is an answer.
        """

        pred_answers = []
        question_captions = samples['question_captions']
        question_captions_chunk = [question_captions[i:i + internal_bsz_fid]
                                   for i in range(0, len(question_captions), internal_bsz_fid)]
        question_captions_chunk = list(chain(*question_captions_chunk))

        for question_caption in question_captions_chunk:
            question_caption_input = self.question_answering.tokenizer(question_caption, padding='longest',
                                                                truncation=True, return_tensors="pt").to(self.device)

            question_caption_input.input_ids = question_caption_input.input_ids.reshape(
                                               internal_bsz_fid, -1, question_caption_input.input_ids.size(1))
            question_caption_input.attention_mask = question_caption_input.attention_mask.reshape(
                                               internal_bsz_fid, -1, question_caption_input.attention_mask.size(1))

            outputs = self.question_answering.generate(input_ids=question_caption_input.input_ids,
                                            attention_mask=question_caption_input.attention_mask,
                                            num_beams=num_beams,
                                            min_length=min_len,
                                            max_length=max_len,
                                            )

            for output in outputs:
                pred_answer = self.question_answering.tokenizer.decode(output, skip_special_tokens=True)
                pred_answers.append(pred_answer)

        return pred_answers

    def predict_answers(
        self,
        samples,
        num_beams=1,
        inference_method="generate",
        max_len=20,
        min_len=0,
        **kwargs
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

        config = kwargs.get('config')

        samples = self.forward_itm(samples, block_num=config['block_num'])

        samples = self.forward_cap(samples,
                                   cap_max_length=config['cap_max_length'],
                                   cap_min_length=config['cap_min_length'],
                                   top_k=config['top_k'],
                                   top_p=config['top_p'],
                                   repetition_penalty=config['repetition_penalty'],
                                   num_captions=config['num_captions'],
                                   num_patches=config['num_patches'])

        prepare_qa_input(samples, num_captions=config['num_captions'], num_captions_fid=config['num_captions_fid'])

        pred_answers = self.forward_qa(samples,
                                  num_beams=num_beams,
                                  max_len=max_len,
                                  min_len=min_len,
                                  internal_bsz_fid=config['internal_bsz_fid'])

        return pred_answers, samples['captions'], samples['gradcams']

    @classmethod
    def from_config(cls, model_config):
        itm_config = model_config.image_question_matching_model
        cap_config = model_config.image_captioning_model
        qa_config = model_config.question_answering_model

        itm_cls = registry.get_model_class(itm_config.arch)
        cap_cls = registry.get_model_class(cap_config.arch)
        qa_cls = registry.get_model_class(qa_config.arch)

        model = cls(itm_cls=itm_cls, cap_cls=cap_cls, qa_cls=qa_cls,
                    itm_config=itm_config, cap_config=cap_config, qa_config=qa_config)

        return model
