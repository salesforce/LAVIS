import os
from urllib.parse import urlparse

import numpy as np
import torch
import torch.nn.functional as F
from common.registry import registry
from omegaconf import OmegaConf
from timm.models.hub import download_cached_file

from models.base_model import EncoderDecoderModel, EncoderEncoderModel
from models.blip import init_tokenizer
from models.med import XBertEncoder, XBertLMHeadDecoder
from models.vit import VisionTransformerEncoder, interpolate_pos_embed

pretrain_specific_keys = set([
    "temp", "image_queue", "text_queue", "queue_ptr",
    "vision_proj.weight", "vision_proj.bias",
    "text_proj.weight", "text_proj.bias",
    "itm_head.weight", "itm_head.bias"]
)


# [TODO] move to utils for reuse
def is_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")


class BlipMultimodalEncoder(EncoderEncoderModel):
    def __init__(self, image_encoder, text_encoder, require_tokenizer=True):
        super().__init__(image_encoder, text_encoder)

        if require_tokenizer:
            self.tokenizer = init_tokenizer()

    def forward_encoder_pre(self, samples):
        image_encoder = self.encoder_pre
        image_embeds = image_encoder(samples['image'])
        
        return image_embeds
    
    def forward_encoder_pst(self, samples, image_enc_out):
        image_embeds = image_enc_out

        multimodal_encoder = self.encoder_pst
        multimodal_embeds = multimodal_encoder(
            tokenized_text=samples['tokenized_questions'],
            visual_embeds=image_embeds
        )

        return image_embeds, multimodal_embeds

    @classmethod
    def build_from_cfg(cls, cfg):
        # vision encoder
        image_encoder = VisionTransformerEncoder.build_from_cfg(cfg)
        vision_width = image_encoder.vision_width
        if "vision_width" not in cfg:
            cfg.vision_width = vision_width

        # text encoder + multimodal encoder
        text_encoder = XBertEncoder.build_from_cfg(cfg)
        model = cls(image_encoder, text_encoder)

        return model


@registry.register_model("blip_caption")
class BlipCaption(EncoderDecoderModel):
    def __init__(self, image_encoder, text_decoder, prompt=None):
        super().__init__(image_encoder, text_decoder)

        self.tokenizer = init_tokenizer()

        self.prompt = prompt
        self.prompt_length = len(self.tokenizer(self.prompt).input_ids) - 1

    @classmethod
    def default_config_path(cls, model_type="base"):
        paths = {
            "base": "configs/models/blip_caption_base.yaml",
            "large": "configs/models/blip_caption_large.yaml"
        }

        assert model_type in paths, "Unknown model type {}".format(model_type)
        return paths[model_type]

    def forward_encoder(self, samples):
        """
        The forward_encoder() and forward_decoder() allows the constituent
        encoder decoder class to be reuse without coupling to a specific vision-language model.

        If instead call encoder(samples), then the forward() definition of
        the constituent encoder has to return in a specific form,
            e.g. {"image_embeds": image_embeds}

        However, in different vision-language models, different return values may be needed.
        In this case, forward_encoder() which bounds to the specific vision-language model, will
        handle this variation.
        """
        image_embeds = self.encoder(samples['image'])

        return image_embeds

    def forward_decoder(self, samples, encoder_out):
        # prepare inputs for forwarding decoder
        raw_text = samples["text_input"]
        text = self.tokenizer(
            raw_text,
            padding='longest',
            truncation=True,
            max_length=40,
            return_tensors="pt"
        ).to(self.device)
        text.input_ids[:, 0] = self.tokenizer.bos_token_id

        image_embeds = encoder_out

        # prepare targets for forwarding decoder
        decoder_targets = text.input_ids.masked_fill(
            text.input_ids == self.tokenizer.pad_token_id, -100
        )
        decoder_targets[:, :self.prompt_length] = -100

        loss, decoder_output = self.decoder.forward_loss(
            text_tokenized=text,
            visual_embeds=image_embeds,
            decoder_targets=decoder_targets
        )

        return {"loss": loss, "decoder_output": decoder_output}

    def generate(self, samples, use_nucleus_sampling=False, num_beams=3, max_length=30, min_length=10, top_p=0.9, repetition_penalty=1.0):
        # prepare inputs for decoder generation.
        encoder_out = self.forward_encoder(samples)
        image_embeds = encoder_out

        prompt = [self.prompt] * image_embeds.size(0)
        prompt = self.tokenizer(prompt, 
            return_tensors="pt").to(self.device)
        prompt.input_ids[:,0] = self.tokenizer.bos_token_id
        prompt.input_ids = prompt.input_ids[:, :-1]

        # get decoded text
        decoder_out = self.decoder.generate_from_encoder(
            tokenized_prompt=prompt,
            visual_embeds=image_embeds,
            sep_token_id=self.tokenizer.sep_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            use_nucleus_sampling=use_nucleus_sampling,
            num_beams=num_beams,
            max_length=max_length,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty
        )

        captions = []
        for output in decoder_out:
            caption = self.tokenizer.decode(output, skip_special_tokens=True)
            captions.append(caption[len(self.prompt):])
        return captions

    @classmethod
    def build_default_model(cls, model_type="base"):
        return cls.build(cfg=None, model_type=model_type)

    @classmethod
    def build(cls, cfg=None, model_type="base"):
        if not cfg:
            # useful when building model without provided configuration file
            from utils.config import Config
            default_config = OmegaConf.load(
                cls.default_config_path(model_type))
            cfg = Config.build_model_config(default_config).model

        return cls._build_from_cfg(cfg)

    @classmethod
    def _build_from_cfg(cls, cfg):
        # vision encoder
        image_encoder = VisionTransformerEncoder.build_from_cfg(cfg)
        if "vision_width" not in cfg:
            cfg.vision_width = image_encoder.vision_width
        # text encoder + multimodal decoder
        text_decoder = XBertLMHeadDecoder.build_from_cfg(cfg)

        prompt = cfg.get("prompt", None)
        model = cls(image_encoder, text_decoder, prompt=prompt)

        # load pre-trained weights
        pretrain_path = cfg.get("pretrained", None)
        if pretrain_path is not None:
            model, msg = cls.load_from_pretrained(model, url_or_filename=pretrain_path)

            assert len(msg.missing_keys) == 0, "Missing keys {}.".format(msg.missing_keys)
            assert len(msg.unexpected_keys) == 0, "Unexpected keys {}.".format(msg.unexpected_keys)

        return model

    @staticmethod
    def load_from_pretrained(model, url_or_filename):
        if is_url(url_or_filename):
            cached_file = download_cached_file(url_or_filename, check_hash=False, progress=True)
            checkpoint = torch.load(cached_file, map_location='cpu')
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location='cpu')
        else:
            raise RuntimeError('checkpoint url or path is invalid')

        state_dict = checkpoint['model']

        if "visual_encoder.pos_embed" in state_dict.keys():
            state_dict['visual_encoder.pos_embed'] = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'], model.encoder)
        elif "encoder.pos_embed" in state_dict.keys():
            state_dict['encoder.pos_embed'] = interpolate_pos_embed(state_dict['encoder.pos_embed'], model.encoder)

        # FIXME rename the keys in pre-trained state_dict() to avoid this hotfix.
        new_state_dict = dict()
        for key in state_dict.keys():
            if key in pretrain_specific_keys:
                continue
            elif "text_encoder" in key:
                continue
            elif "_m" in key:
                continue
            elif "visual_encoder" in key:
                new_key = key.replace("visual_encoder", "encoder")
            elif "text_decoder" in key:
                new_key = key.replace("text_decoder", "decoder")
            else:
                new_key = key
            new_state_dict[new_key] = state_dict[key]

        # update old state_dict
        state_dict = new_state_dict

        # exclude incompatible keys
        for key in model.state_dict().keys():
            if key in state_dict.keys():
                if state_dict[key].shape != model.state_dict()[key].shape:
                    del state_dict[key]

        msg = model.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % url_or_filename)
        return model, msg


@registry.register_model("blip_vqa")
class BlipVQA(EncoderDecoderModel):
    def __init__(self, image_encoder, text_encoder, text_decoder):
        """
        """
        multimodal_encoder = BlipMultimodalEncoder(
            image_encoder=image_encoder,
            text_encoder=text_encoder,
            require_tokenizer=False
        )

        super().__init__(encoder=multimodal_encoder, decoder=text_decoder)
        self.tokenizer = init_tokenizer()

    @classmethod
    def default_config_path(cls, model_type="base"):
        paths = {
            "base": "configs/models/blip_vqa_base.yaml",
            # "large": "configs/models/blip_vqa_large.yaml"
        }

        assert model_type in paths, "Unknown model type {}".format(model_type)
        return paths[model_type]

    def forward_encoder(self, samples):
        """
        The forward_encoder() and forward_decoder() allows the constituent
        encoder decoder class to be reuse without coupling to a specific vision-language model.

        If instead call encoder(samples), then the forward() definition of
        the constituent encoder has to return in a specific form,
            e.g. {"image_embeds": image_embeds}

        However, in different vision-language models, different return values may be needed.
        In this case, forward_encoder() which bounds to the specific vision-language model, will
        handle this variation.

        """
        questions = samples['question']
        questions = self.tokenizer(
            questions,
            padding='longest',
            truncation=True,
            max_length=35,
            return_tensors="pt").to(self.device)
        questions.input_ids[:, 0] = self.tokenizer.enc_token_id
        samples.update({'tokenized_questions': questions})

        image_embeds, multimodal_embeds = self.encoder(samples)
        return multimodal_embeds


    def forward_decoder(self, samples, encoder_out, **kwargs):
        answers = self.tokenizer(samples["answer"], padding="longest", return_tensors="pt").to(self.device)
        answers.input_ids[:,0] = self.tokenizer.bos_token_id
        answer_targets = answers.input_ids.masked_fill(answers.input_ids == self.tokenizer.pad_token_id, -100)

        question_states = []
        question_atts = []

        question = samples["tokenized_questions"]
        question_output = encoder_out

        for b, n in enumerate(samples["n_answers"]):
            question_states += [question_output.last_hidden_state[b]] * n
            question_atts += [question.attention_mask[b]] * n

        question_states = torch.stack(question_states, dim=0)
        question_atts = torch.stack(question_atts, dim=0)

        answer_output = self.decoder(
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

        return {"loss": loss}

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
                 samples, 
                 num_beams=num_beams, 
                 max_length=max_len, 
                 min_length=min_len
            )
        else:
            return self.rank_answers(
                samples,
                answer_list=answer_list,
                num_ans_candidates=num_ans_candidates
            )

    def generate_answers(self, samples, num_beams=3, max_length=10, min_length=1):
        encoder_out = self.forward_encoder(samples)

        question_output = encoder_out

        question_states = question_output.last_hidden_state.repeat_interleave(num_beams, dim=0)
        question_atts = torch.ones(
            question_states.size()[:-1],
            dtype=torch.long
            ).to(self.device)

        model_kwargs = {
            "encoder_hidden_states": question_states,
            "encoder_attention_mask": question_atts
        }

        bsz = samples["image"].size(0)
        bos_ids = torch.full(
            (bsz, 1),
            fill_value=self.tokenizer.bos_token_id,
            device=self.device
        )

        outputs = self.decoder.generate(
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
        answer_candidates = self.tokenizer(answer_list, padding='longest', return_tensors='pt').to(self.device)
        answer_candidates.input_ids[:,0] = self.tokenizer.bos_token_id

        answer_ids = answer_candidates.input_ids
        answer_atts = answer_candidates.attention_mask

        question_output = self.forward_encoder(samples)
        question_states = question_output.last_hidden_state

        tokenized_question = samples['tokenized_questions']
        question_atts = tokenized_question.attention_mask

        num_ques = question_states.size(0)
        start_ids = answer_ids[0, 0].repeat(num_ques,1) # bos token
        
        start_output = self.decoder(
            start_ids,
            encoder_hidden_states=question_states,
            encoder_attention_mask=question_atts,
            return_dict=True,
            reduction='none'
        )
        logits = start_output.logits[:, 0, :] # first token's logit
        
        # topk_probs: top-k probability 
        # topk_ids: [num_question, k]        
        answer_first_token = answer_ids[:, 1]
        prob_first_token = F.softmax(logits,dim=1).index_select(dim=1, index=answer_first_token) 
        topk_probs, topk_ids = prob_first_token.topk(num_ans_candidates, dim=1) 
        
        # answer input: [num_question*k, answer_len]                 
        input_ids = []
        input_atts = []
        for b, topk_id in enumerate(topk_ids):
            input_ids.append(answer_ids.index_select(dim=0, index=topk_id))
            input_atts.append(answer_atts.index_select(dim=0, index=topk_id))
        input_ids = torch.cat(input_ids,dim=0)  
        input_atts = torch.cat(input_atts,dim=0)  

        targets_ids = input_ids.masked_fill(input_ids == self.tokenizer.pad_token_id, -100)

        # repeat encoder's output for top-k answers
        question_states = tile(question_states, 0, num_ans_candidates)
        question_atts = tile(question_atts, 0, num_ans_candidates)
        
        output = self.decoder(
            input_ids,
            attention_mask=input_atts,
            encoder_hidden_states=question_states,
            encoder_attention_mask=question_atts,
            labels=targets_ids,
            return_dict=True,
            reduction="none"
        )
        
        log_probs_sum = -output.loss
        log_probs_sum = log_probs_sum.view(num_ques, num_ans_candidates)

        max_topk_ids = log_probs_sum.argmax(dim=1) 
        max_ids = topk_ids[max_topk_ids>=0, max_topk_ids]

        answers = [answer_list[max_id] for max_id in max_ids]

        return answers


    @classmethod
    def build_default_model(cls, model_type="base"):
        return cls.build(cfg=None, model_type=model_type)


    @classmethod
    def build(cls, cfg=None, model_type="base"):
        if not cfg:
            # useful when building model without provided configuration file
            from utils.config import Config
            default_config = OmegaConf.load(cls.default_config_path(model_type))
            cfg = Config.build_model_config(default_config).model

        return cls._build_from_cfg(cfg)


    @classmethod
    def _build_from_cfg(cls, cfg=None):
        image_encoder = VisionTransformerEncoder.build_from_cfg(cfg)
        vision_width = image_encoder.vision_width
        if "vision_width" not in cfg:
            cfg.vision_width = vision_width

        # text encoder + multimodal encoder
        text_encoder = XBertEncoder.build_from_cfg(cfg)
        text_decoder = XBertLMHeadDecoder.build_from_cfg(cfg)

        model = cls(
            image_encoder=image_encoder,
            text_encoder=text_encoder,
            text_decoder=text_decoder
        )

        # load pre-trained weights
        pretrain_path = cfg.get("pretrained", None)
        if pretrain_path is not None:
            model, msg = cls.load_from_pretrained(model, url_or_filename=pretrain_path)

            assert len(msg.missing_keys) == 0, "Missing keys {}.".format(msg.missing_keys)
            assert len(msg.unexpected_keys) == 0, "Unexpected keys {}.".format(msg.unexpected_keys)

        return model

    @staticmethod
    def load_from_pretrained(model, url_or_filename):
        if is_url(url_or_filename):
            cached_file = download_cached_file(url_or_filename, check_hash=False, progress=True)
            checkpoint = torch.load(cached_file, map_location='cpu')
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location='cpu')
        else:
            raise RuntimeError('checkpoint url or path is invalid')

        state_dict = checkpoint['model']

        if "visual_encoder.pos_embed" in state_dict.keys():
            state_dict['visual_encoder.pos_embed'] = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'], model.encoder.encoder_pre)
        elif "encoder.pos_embed" in state_dict.keys():
            state_dict['encoder.pos_embed'] = interpolate_pos_embed(state_dict['encoder.pos_embed'], model.encoder)

        # FIXME rename the keys in pre-trained state_dict() to avoid this hotfix.
        new_state_dict = dict()
        for key in state_dict.keys():
            if key in pretrain_specific_keys:
                continue
            elif "_m" in key:
                continue
            elif "visual_encoder" in key:
                new_key = key.replace("visual_encoder", "encoder.encoder_pre")
            elif "text_encoder" in key:
                new_key = key.replace("text_encoder", "encoder.encoder_pst")
            elif "text_decoder" in key:
                new_key = key.replace("text_decoder", "decoder")
            else:
                new_key = key
            new_state_dict[new_key] = state_dict[key]

        # update old state_dict
        state_dict = new_state_dict

        # exclude incompatible keys
        for key in model.state_dict().keys():
            if key in state_dict.keys():
                if state_dict[key].shape!=model.state_dict()[key].shape:
                    del state_dict[key]

        msg = model.load_state_dict(state_dict,strict=False)
        print('load checkpoint from %s'%url_or_filename)
        return model, msg


def tile(x, dim, n_tile):
    init_dim = x.size(dim)
    repeat_idx = [1] * x.dim()
    repeat_idx[dim] = n_tile
    x = x.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(x, dim, order_index.to(x.device))


# @registry.register_model("blip_retrieval")
# class BlipRetrieval(BlipEncoderEncoder):
#     def __init__(self, image_encoder, text_encoder, embed_dim):
#         super().__init__(image_encoder, text_encoder)

#         vision_width = image_encoder.vision_width
#         self.vision_proj = nn.Linear(vision_width, embed_dim)

#         text_width = text_encoder.config.hidden_size 
#         self.text_proj = nn.Linear(text_width, embed_dim)

#         self.itm_head = nn.Linear(text_width, 2) 

#     @classmethod
#     def default_config_path(cls, model_type="base"):
#         paths = {
#             "base": "configs/models/blip_enc_enc_base.yaml",
#             "large": "configs/models/blip_enc_enc_large.yaml"
#         }

#         assert model_type in paths, "Unknown model type {}".format(model_type)
#         return paths[model_type]

#     def forward_encoder_pre(self, samples):
#         """
#         The forward_encoder() and forward_decoder() allows the constituent
#         encoder decoder class to be reuse without coupling to a specific vision-language model.

#         If instead call encoder(samples), then the forward() definition of
#         the constituent encoder has to return in a specific form, 
#             e.g. {"image_embeds": image_embeds}

#         However, in different vision-language models, different return values may be needed.
#         In this case, forward_encoder() which bounds to the specific vision-language model, will 
#         handle this variation.
        
#         """
#         return {'image_embeds': self.encoder_pre(samples['vis_input'])}
    
#     def forward_encoder_pst(self, samples, encoder_pre_out, **kwargs):
#         pass

#     @classmethod
#     def build_from_cfg(cls, cfg=None, model_type="base"):
#         if not cfg:
#             # useful when building model without provided configuration file
#             from utils.config import Config
#             cfg = Config.build_model_config(config_path=cls.default_config_path(model_type)).model
        
#         return cls._build_from_cfg(cfg)
    
#     @classmethod
#     def _build_from_cfg(cls, cfg):
#         embed_dim = cfg.get("embed_dim", 256)

#         # vision encoder
#         encoder_vis = VisionTransformerEncoder.build_from_cfg(cfg) 
#         vision_width = encoder_vis.vision_width
#         if "vision_width" not in cfg:
#             cfg.vision_width = vision_width

#         # text encoder + multimodal encoder
#         encoder_xmodal = XBertEncoder.build_from_cfg(cfg)
#         model = cls(encoder_vis, encoder_xmodal, embed_dim=embed_dim)

#         # load pre-trained weights
#         pretrain_path = cfg.get("pretrained", None)
#         if pretrain_path is not None:
#             model, msg = cls.load_from_pretrained(model, url_or_filename=pretrain_path)
        
#             assert len(msg.missing_keys) == 0, "Missing keys {}.".format(msg.missing_keys)
#             assert len(msg.unexpected_keys) == 0, "Unexpected keys {}.".format(msg.unexpected_keys)

#         return model 

#     @staticmethod
#     def load_from_pretrained(model, url_or_filename):
#         raise NotImplementedError
#         # # [TODO] move to utils for reuse
#         # if is_url(url_or_filename):
#         #     cached_file = download_cached_file(url_or_filename, check_hash=False, progress=True)
#         #     checkpoint = torch.load(cached_file, map_location='cpu') 
#         # elif os.path.isfile(url_or_filename):        
#         #     checkpoint = torch.load(url_or_filename, map_location='cpu') 
#         # else:
#         #     raise RuntimeError('checkpoint url or path is invalid')

#         # state_dict = checkpoint['model']
        
#         # if "visual_encoder.pos_embed" in state_dict.keys():
#         #     state_dict['visual_encoder.pos_embed'] = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'], model.encoder) 
#         # elif "encoder.pos_embed" in state_dict.keys():
#         #     state_dict['encoder.pos_embed'] = interpolate_pos_embed(state_dict['encoder.pos_embed'], model.encoder) 

#         # pretrain_specific_keys = set([
#         #     "temp", "image_queue", "text_queue", "queue_ptr", 
#         #     "vision_proj.weight", "vision_proj.bias",
#         #     "text_proj.weight", "text_proj.bias",
#         #     "itm_head.weight", "itm_head.bias"]
#         # )
#         # # FIXME rename the keys in pre-trained state_dict() to avoid this hotfix.
#         # new_state_dict = dict()
#         # for key in state_dict.keys():
#         #     if key in pretrain_specific_keys:
#         #         continue
#         #     elif "text_encoder" in key:
#         #         continue
#         #     elif "_m" in key:
#         #         continue
#         #     elif "visual_encoder" in key:
#         #         new_key = key.replace("visual_encoder", "encoder")
#         #     elif "text_decoder" in key:
#         #         new_key = key.replace("text_decoder", "decoder")
#         #     else:
#         #         new_key = key
#         #     new_state_dict[new_key] = state_dict[key]

#         # # update old state_dict
#         # state_dict = new_state_dict

#         # # exclude incompatible keys
#         # for key in model.state_dict().keys():
#         #     if key in state_dict.keys():
#         #         if state_dict[key].shape!=model.state_dict()[key].shape:
#         #             del state_dict[key]
        
#         # msg = model.load_state_dict(state_dict,strict=False)
#         # print('load checkpoint from %s'%url_or_filename)  
#         # return model, msg
    