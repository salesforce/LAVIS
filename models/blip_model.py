import os

import numpy as np
import torch
import torch.nn.functional as F

from copy import deepcopy

from torch import nn
from common.registry import registry
from timm.models.hub import download_cached_file

from models.base_model import BaseModel
from models.blip import init_tokenizer, is_url
from models.med import XBertEncoder, XBertLMHeadDecoder
from models.vit import VisionTransformerEncoder, interpolate_pos_embed

from utils.blip_utils import tie_encoder_decoder_weights

pretrain_specific_keys = set(
    [
        "temp", "image_queue", "text_queue", "queue_ptr",
        "vision_proj.weight", "vision_proj.bias",
        "text_proj.weight", "text_proj.bias",
        "itm_head.weight", "itm_head.bias",
        "vision_proj_m.weight", "vision_proj_m.bias",
        "text_proj_m.weight", "text_proj_m.bias"
    ]
)


class MomentumDistilationMixin:
    @torch.no_grad()    
    def copy_params(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient    


    @torch.no_grad()        
    def _momentum_update(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)


@registry.register_model("blip_caption")
class BlipCaption(BaseModel):
    def __init__(self, image_encoder, text_decoder, prompt=None):
        super().__init__()

        self.tokenizer = init_tokenizer()

        self.visual_encoder = image_encoder
        self.text_decoder = text_decoder

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
        image_embeds = self.visual_encoder(samples['image'])
        return image_embeds

    def forward_decoder(self, samples, image_embeds):
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

        # prepare targets for forwarding decoder
        decoder_targets = text.input_ids.masked_fill(
            text.input_ids == self.tokenizer.pad_token_id, -100
        )
        decoder_targets[:, :self.prompt_length] = -100

        _, decoder_output = self.text_decoder.forward_loss(
            text_tokenized=text,
            visual_embeds=image_embeds,
            decoder_targets=decoder_targets
        )

        return {k:decoder_output[k] for k in decoder_output}
    
    def forward(self, samples):
        image_embeds = self.forward_encoder(samples)
        decoder_out = self.forward_decoder(samples, image_embeds)

        return decoder_out

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
        decoder_out = self.text_decoder.generate_from_encoder(
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
    def _build_from_cfg(cls, cfg):
        # vision encoder
        image_encoder = VisionTransformerEncoder.build_from_cfg(cfg)
        # text encoder + multimodal decoder
        text_decoder = XBertLMHeadDecoder.build_from_cfg(cfg)

        prompt = cfg.get("prompt", None)
        model = cls(image_encoder, text_decoder, prompt=prompt)

        # load pre-trained weights
        pretrain_path = cfg.get("pretrained", None)
        if pretrain_path is not None:
            model, msg = load_from_pretrained(model, url_or_filename=pretrain_path)

        return model


@registry.register_model("blip_vqa")
class BlipVQA(BaseModel):
    def __init__(self, image_encoder, text_encoder, text_decoder):
        super().__init__()
        self.tokenizer = init_tokenizer()

        self.visual_encoder = image_encoder

        self.text_encoder = text_encoder
        self.text_decoder = text_decoder

    @classmethod
    def default_config_path(cls, model_type="base"):
        paths = {
            "base": "configs/models/blip_vqa_base.yaml",
            # "large": "configs/models/blip_vqa_large.yaml"
        }

        assert model_type in paths, "Unknown model type {}".format(model_type)
        return paths[model_type]

    def forward(self, samples):
        multimodal_embeds = self.forward_encoder(samples)
        decoder_out = self.forward_decoder(samples, encoder_out=multimodal_embeds)

        return decoder_out

    def forward_encoder(self, samples):
        # TODO rename to 'text_input'?
        questions = samples['question']
        questions = self.tokenizer(
            questions,
            padding='longest',
            truncation=True,
            max_length=35,
            return_tensors="pt").to(self.device)
        questions.input_ids[:, 0] = self.tokenizer.enc_token_id
        samples.update({'tokenized_text': questions})

        image_embeds = self.visual_encoder(samples["image"])
        multimodal_embeds = self.text_encoder(
            tokenized_text=samples['tokenized_text'],
            visual_embeds=image_embeds
        )

        return multimodal_embeds


    def forward_decoder(self, samples, encoder_out, **kwargs):
        answers = self.tokenizer(samples["answer"], padding="longest", return_tensors="pt").to(self.device)
        answers.input_ids[:,0] = self.tokenizer.bos_token_id
        answer_targets = answers.input_ids.masked_fill(answers.input_ids == self.tokenizer.pad_token_id, -100)

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
        answer_candidates = self.tokenizer(answer_list, padding='longest', return_tensors='pt').to(self.device)
        answer_candidates.input_ids[:,0] = self.tokenizer.bos_token_id

        answer_ids = answer_candidates.input_ids
        answer_atts = answer_candidates.attention_mask

        question_output = self.forward_encoder(samples)
        question_states = question_output.last_hidden_state

        tokenized_question = samples['tokenized_text']
        question_atts = tokenized_question.attention_mask

        num_ques = question_states.size(0)
        start_ids = answer_ids[0, 0].repeat(num_ques,1) # bos token
        
        start_output = self.text_decoder(
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
        
        output = self.text_decoder(
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
    def _build_from_cfg(cls, cfg=None):
        image_encoder = VisionTransformerEncoder.build_from_cfg(cfg)

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
            model, msg = load_from_pretrained(model, url_or_filename=pretrain_path)

        return model


@registry.register_model("blip_classification")
class BlipClassification(BaseModel, MomentumDistilationMixin):
    def __init__(
        self, 
        image_encoder,
        text_encoder,
        num_classes,
        momentum=0.995,
        alpha=0.4,
        use_distill=True
    ):
        super().__init__()

        self.tokenizer = init_tokenizer()

        self.use_distill = use_distill

        self.visual_encoder = image_encoder
        self.text_encoder = text_encoder

        hidden_size = text_encoder.config.hidden_size
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

        if self.use_distill:
            self.visual_encoder_m = deepcopy(self.visual_encoder)
            self.text_encoder_m = deepcopy(self.text_encoder)
            self.cls_head_m = deepcopy(self.cls_head)

            self.momentum = momentum
            self.alpha = alpha

            self.model_pairs = [
                [self.visual_encoder, self.visual_encoder_m],
                [self.text_encoder, self.text_encoder_m],
                [self.cls_head, self.cls_head_m],
            ]

            self.copy_params()

    @classmethod
    def default_config_path(cls, model_type="base"):
        paths = {
            "base": "configs/models/blip_ve_base.yaml",
            # "large": "configs/models/blip_pretrain_large.yaml"
        }

        assert model_type in paths, "Unknown model type {}".format(model_type)
        return paths[model_type]

    def _rampup_factor(self, epoch, iters, num_iters_per_epoch):
        return min(1, (epoch * num_iters_per_epoch + iters) / num_iters_per_epoch)

    def forward(self, samples, is_train=True):
        sentences = samples['text_input']
        sentences = self.tokenizer(
            sentences, 
            padding="longest",
            return_tensors="pt"
        ).to(self.device)
        samples.update({'tokenized_text': sentences})

        targets = samples['label']

        image_embeds = self.visual_encoder(samples['image'])
        multimodal_embeds = self.text_encoder(samples['tokenized_text'], image_embeds)

        prediction = self.cls_head(multimodal_embeds.last_hidden_state[:,0,:])

        if is_train:
            if self.use_distill:
                with torch.no_grad():
                    self._momentum_update()

                    image_embeds_m = self.visual_encoder_m(samples['image'])
                    multimodal_embeds_m = self.text_encoder_m(samples['tokenized_text'], image_embeds_m)

                    prediction_m = self.cls_head_m(
                        multimodal_embeds_m.last_hidden_state[:,0,:]
                    )

                alpha = self.alpha * self._rampup_factor(
                    epoch=samples['epoch'],
                    iters=samples['iters'],
                    num_iters_per_epoch=samples['num_iters_per_epoch']
                )

                loss = (1 - alpha) * F.cross_entropy(prediction, targets) - alpha * torch.sum(
                    F.log_softmax(prediction, dim=1) * F.softmax(prediction_m, dim=1),dim=1).mean()
            else:
                loss = F.cross_entropy(prediction, targets)

            return {"loss": loss}
        else:
            return {"predictions": prediction, "targets": targets}

    def predict(self, samples):
        output = self.forward(samples, is_train=False)
        return output

    @classmethod
    def _build_from_cfg(cls, cfg=None):
        image_encoder = VisionTransformerEncoder.build_from_cfg(cfg)

        # text encoder + multimodal encoder
        text_encoder = XBertEncoder.build_from_cfg(cfg)
        use_distill = cfg.get("use_distill", True)
        momentum = cfg.get("momentum", 0.995)
        num_classes = cfg.get("num_classes", -1)
        alpha = cfg.get("alpha", 0.4)

        assert num_classes > 1, "Invalid number of classes provided, found {}".format(num_classes)

        model = cls(
            image_encoder=image_encoder,
            text_encoder=text_encoder,
            use_distill=use_distill,
            alpha=alpha,
            num_classes=num_classes,
            momentum=momentum
        )

        # load pre-trained weights
        pretrain_path = cfg.get("pretrained", None)
        if pretrain_path is not None:
            model, msg = load_from_pretrained(model, url_or_filename=pretrain_path)

        return model


@registry.register_model('blip_pretrain')
class BlipPretrain(BaseModel, MomentumDistilationMixin):
    def __init__(
        self, 
        image_encoder,
        text_encoder,
        text_decoder,
        queue_size,
        alpha=0.4,
        embed_dim=256,
        momentum=0.995,
        tie_enc_dec_weights=True
    ):
        """
        """
        super().__init__()

        self.tokenizer = init_tokenizer()

        text_encoder.resize_token_embeddings(len(self.tokenizer))
        text_decoder.resize_token_embeddings(len(self.tokenizer))

        if tie_enc_dec_weights:
            tie_encoder_decoder_weights(
                encoder=text_encoder, 
                decoder=text_decoder.bert,
                base_model_prefix='',
                skip_key="/attention"
            )

        self.visual_encoder = image_encoder

        self.text_encoder = text_encoder
        self.text_decoder = text_decoder

        # creating projection layers for ITC
        text_width = text_encoder.config.hidden_size
        vision_width = image_encoder.vision_width

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
        self.temp = nn.Parameter(0.07 * torch.ones([]))   

        self.alpha = alpha


    @classmethod
    def default_config_path(cls, model_type="base"):
        paths = {
            "base": "configs/models/blip_pretrain_base.yaml",
            "large": "configs/models/blip_pretrain_large.yaml"
        }

        assert model_type in paths, "Unknown model type {}".format(model_type)
        return paths[model_type]

    def _rampup_factor(self, epoch, iters, num_iters_per_epoch):
        return min(1, (epoch * num_iters_per_epoch + iters) / (2 * num_iters_per_epoch))

    def forward(self, samples):
        image = samples['image']
        caption = samples['text_input']

        alpha = self.alpha * self._rampup_factor(
            epoch=samples['epoch'],
            iters=samples['iters'],
            num_iters_per_epoch=samples['num_iters_per_epoch']
        )

        with torch.no_grad():
            self.temp.clamp_(0.001,0.5)

        image_embeds = self.visual_encoder(image) 
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)        
        image_feat = F.normalize(self.vision_proj(image_embeds[:,0,:]),dim=-1)          
        
        text = self.tokenizer(
            caption, 
            padding='max_length',
            truncation=True,
            max_length=30, 
            return_tensors="pt"
        ).to(image.device)  

        text_output = self.text_encoder.forward_text_embeds(text)
        text_feat = F.normalize(self.text_proj(text_output.last_hidden_state[:,0,:]), dim=-1)
            
        # get momentum features
        with torch.no_grad():
            self._momentum_update()
            image_embeds_m = self.visual_encoder_m(image) 
            image_feat_m = F.normalize(self.vision_proj_m(image_embeds_m[:,0,:]),dim=-1)  
            image_feat_all = torch.cat([image_feat_m.t(),self.image_queue.clone().detach()],dim=1)                   
            
            text_output_m = self.text_encoder_m.forward_text_embeds(text)
            text_feat_m = F.normalize(self.text_proj_m(text_output_m.last_hidden_state[:,0,:]),dim=-1) 
            text_feat_all = torch.cat([text_feat_m.t(),self.text_queue.clone().detach()],dim=1)

            sim_i2t_m = image_feat_m @ text_feat_all / self.temp  
            sim_t2i_m = text_feat_m @ image_feat_all / self.temp 

            sim_targets = torch.zeros(sim_i2t_m.size()).to(image.device)
            sim_targets.fill_diagonal_(1)          

            sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
            sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets        

        sim_i2t = image_feat @ text_feat_all / self.temp
        sim_t2i = text_feat @ image_feat_all / self.temp
                            
        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*sim_i2t_targets,dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*sim_t2i_targets,dim=1).mean() 

        loss_ita = (loss_i2t+loss_t2i)/2

        self._dequeue_and_enqueue(image_feat_m, text_feat_m)        

        ###============== Image-text Matching ===================###
        encoder_input_ids = text.input_ids.clone()
        encoder_input_ids[:,0] = self.tokenizer.enc_token_id
        
        # forward the positve image-text pair
        bs = image.size(0)
        output_pos = self.text_encoder.forward(
            tokenized_text=text,
            visual_embeds=image_embeds
        )

        with torch.no_grad():       
            weights_t2i = F.softmax(sim_t2i[:,:bs],dim=1)+1e-4 
            weights_t2i.fill_diagonal_(0)            
            weights_i2t = F.softmax(sim_i2t[:,:bs],dim=1)+1e-4  
            weights_i2t.fill_diagonal_(0)   
            
        # select a negative image for each text
        image_embeds_neg = []    
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeds_neg.append(image_embeds[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg,dim=0)   

        # select a negative text for each image
        text_ids_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_ids_neg.append(encoder_input_ids[neg_idx])
            text_atts_neg.append(text.attention_mask[neg_idx])

        text_ids_neg = torch.stack(text_ids_neg,dim=0)   
        text_atts_neg = torch.stack(text_atts_neg,dim=0)      

        text_ids_all = torch.cat([encoder_input_ids, text_ids_neg],dim=0)     
        text_atts_all = torch.cat([text.attention_mask, text_atts_neg],dim=0)     

        image_embeds_all = torch.cat([image_embeds_neg,image_embeds],dim=0)
        image_atts_all = torch.cat([image_atts,image_atts],dim=0)

        output_neg = self.text_encoder.forward_bert(
            text_ids_all,
            attention_mask = text_atts_all,
            encoder_hidden_states = image_embeds_all,
            encoder_attention_mask = image_atts_all,      
            return_dict=True,
        )                            

        vl_embeddings = torch.cat([output_pos.last_hidden_state[:,0,:], output_neg.last_hidden_state[:,0,:]],dim=0)
        vl_output = self.itm_head(vl_embeddings)            

        itm_labels = torch.cat([torch.ones(bs,dtype=torch.long),torch.zeros(2*bs,dtype=torch.long)],
                            dim=0).to(image.device)
        loss_itm = F.cross_entropy(vl_output, itm_labels)  
        
        ##================= LM ========================##     
        decoder_input_ids = text.input_ids.clone()      
        decoder_input_ids[:,0] = self.tokenizer.bos_token_id
        decoder_targets = decoder_input_ids.masked_fill(decoder_input_ids == self.tokenizer.pad_token_id, -100) 

        loss_lm, decoder_output = self.text_decoder.forward_loss(
            text_tokenized=text,
            visual_embeds=image_embeds,
            decoder_targets=decoder_targets
        )   
        
        return {
            "loss": loss_ita + loss_itm + loss_lm,
            "loss_ita": loss_ita,
            "loss_itm": loss_itm,
            "loss_lm": loss_lm
        }

    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat):
        # gather keys before updating queue
        image_feats = concat_all_gather(image_feat)
        text_feats = concat_all_gather(text_feat)

        batch_size = image_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr 


    @classmethod
    def _build_from_cfg(cls, cfg=None):
        # set from_pretrained=True to load weights for 'bert-base-uncased'
        image_encoder = VisionTransformerEncoder.build_from_cfg(cfg, from_pretrained=True)
        text_encoder = XBertEncoder.build_from_cfg(cfg, from_pretrained=True)
        text_decoder = XBertLMHeadDecoder.build_from_cfg(cfg, from_pretrained=True)

        embed_dim = cfg.get("embed_dim", 256)
        momentum = cfg.get("momentum", 0.995)
        alpha = cfg.get("alpha", 0.4)
        queue_size = cfg.get("queue_size", None)

        assert queue_size, "queue_size must be specified."

        model = cls(
            image_encoder=image_encoder,
            text_encoder=text_encoder,
            text_decoder=text_decoder,
            embed_dim=embed_dim,
            queue_size=queue_size,
            momentum=momentum,
            alpha=alpha,
            tie_enc_dec_weights=True
        )

        return model


def tile(x, dim, n_tile):
    init_dim = x.size(dim)
    repeat_idx = [1] * x.dim()
    repeat_idx[dim] = n_tile
    x = x.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(x, dim, order_index.to(x.device))


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output     


def load_from_pretrained(model, url_or_filename):
    if is_url(url_or_filename):
        cached_file = download_cached_file(url_or_filename, check_hash=False, progress=True)
        checkpoint = torch.load(cached_file, map_location='cpu')
    elif os.path.isfile(url_or_filename):
        checkpoint = torch.load(url_or_filename, map_location='cpu')
    else:
        raise RuntimeError('checkpoint url or path is invalid')

    state_dict = checkpoint['model']

    state_dict['visual_encoder.pos_embed'] = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'], model.visual_encoder)
    if 'visual_encoder_m.pos_embed' in model.state_dict().keys():
        state_dict['visual_encoder_m.pos_embed'] = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'], model.visual_encoder_m)

    for key in model.state_dict().keys():
        if key in state_dict.keys():
            if state_dict[key].shape!=model.state_dict()[key].shape:
                del state_dict[key]

    msg = model.load_state_dict(state_dict, strict=False)
    print('load checkpoint from %s' % url_or_filename)
    return model, msg


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
    