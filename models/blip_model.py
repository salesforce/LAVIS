import os
from urllib.parse import urlparse
from omegaconf import OmegaConf

import torch
from torch import nn
from common.registry import registry
from timm.models.hub import download_cached_file

from models.base_model import EncoderDecoderModel, EncoderEncoderModel
from models.blip import init_tokenizer
from models.med import XBertEncoder, XBertLMHeadDecoder
from models.vit import VisionTransformerEncoder, interpolate_pos_embed


# [TODO] move to utils for reuse
def is_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")


@registry.register_model("blip_enc_dec")
class BlipEncoderDecoder(EncoderDecoderModel):
    def __init__(self, encoder, decoder, prompt=None):
        super().__init__(encoder, decoder)

        self.tokenizer = init_tokenizer()

        self.prompt = prompt 
        self.prompt_length = len(self.tokenizer(self.prompt).input_ids) - 1
        
    @classmethod
    def default_config_path(cls, model_type="base"):
        paths = {
            "base": "configs/models/blip_enc_dec_base.yaml",
            "large": "configs/models/blip_enc_dec_large.yaml"
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
        return {'image_embeds': self.encoder(samples['vis_data'])}
    
    def forward_decoder(self, samples, encoder_out, **kwargs):
        image_embeds = encoder_out["image_embeds"]
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(self.device)

        raw_text = samples["text_data"]
        text = self.tokenizer(
            raw_text, 
            padding='longest',
            truncation=True,
            max_length=40, 
            return_tensors="pt"
        ).to(self.device)

        text.input_ids[:, 0] = self.tokenizer.bos_token_id
        
        decoder_targets = text.input_ids.masked_fill(text.input_ids==self.tokenizer.pad_token_id, -100)         
        decoder_targets[:,:self.prompt_length] = -100
     
        decoder_output = self.decoder(text.input_ids,
                                      attention_mask=text.attention_mask, 
                                      encoder_hidden_states=image_embeds,
                                      encoder_attention_mask=image_atts,                  
                                      labels=decoder_targets,
                                      return_dict=True,   
                                    )   
        loss_lm = decoder_output.loss
        
        return loss_lm

    def generate(self, samples, use_nucleus_sampling=False, num_beams=3, max_length=30, min_length=10, top_p=0.9, repetition_penalty=1.0):
        # get image embeddings
        enc_out = self.forward_encoder(samples)

        # prepare model output for decoder
        image_embeds = enc_out['image_embeds']

        prompt = [self.prompt] * image_embeds.size(0)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(image_embeds.device) 
        input_ids[:,0] = self.tokenizer.bos_token_id
        input_ids = input_ids[:, :-1] 

        if not use_nucleus_sampling:
            num_beams = num_beams
            image_embeds = image_embeds.repeat_interleave(num_beams, dim=0)

            enc_out['image_embeds'] = image_embeds

        # get decoded text 
        decoder_out = self.decoder.generate_from_visual(
            input_ids=input_ids,
            max_length=max_length,
            min_length=min_length,
            do_sample=use_nucleus_sampling,
            top_p=top_p,
            tokenizer=self.tokenizer,
            repetition_penalty=repetition_penalty,
            num_beams=num_beams,
            **enc_out
        )

        captions = []    
        for output in decoder_out:
            caption = self.tokenizer.decode(output, skip_special_tokens=True)    
            captions.append(caption[len(self.prompt):])
        return captions

    @classmethod
    def build_default_model(cls, model_type="base"):
        return cls.build_model(cfg=None, model_type=model_type)

    @classmethod
    def build_model(cls, cfg=None, model_type="base"):
        if not cfg:
            # useful when building model without provided configuration file
            from utils.config import Config
            default_config = OmegaConf.load(cls.default_config_path(model_type))
            cfg = Config.build_model_config(default_config).model
        
        return cls._build_model_from_config(cfg)
    
    @classmethod
    def _build_model_from_config(cls, cfg):
        # vision encoder
        encoder = VisionTransformerEncoder.build_model(cfg) 
        if "vision_width" not in cfg:
            cfg.vision_width = encoder.vision_width
        # text encoder + multimodal decoder
        decoder = XBertLMHeadDecoder.build_model(cfg)

        prompt = cfg.get("prompt", None)
        model = cls(encoder, decoder, prompt=prompt)

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

        pretrain_specific_keys = set([
            "temp", "image_queue", "text_queue", "queue_ptr", 
            "vision_proj.weight", "vision_proj.bias",
            "text_proj.weight", "text_proj.bias",
            "itm_head.weight", "itm_head.bias"]
        )
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
                if state_dict[key].shape!=model.state_dict()[key].shape:
                    del state_dict[key]
        
        msg = model.load_state_dict(state_dict,strict=False)
        print('load checkpoint from %s'%url_or_filename)  
        return model, msg
    

@registry.register_model("blip_enc_enc")
class BlipEncoderEncoder(EncoderEncoderModel):
    def __init__(self, encoder_pre, encoder_pst, embed_dim):
        super().__init__(encoder_pre, encoder_pst)

        self.tokenizer = init_tokenizer()

        vision_width = encoder_pre.vision_width
        self.vision_proj = nn.Linear(vision_width, embed_dim)

        text_width = encoder_pst.config.hidden_size 
        self.text_proj = nn.Linear(text_width, embed_dim)

        self.itm_head = nn.Linear(text_width, 2) 

    @classmethod
    def default_config_path(cls, model_type="base"):
        paths = {
            "base": "configs/models/blip_enc_enc_base.yaml",
            "large": "configs/models/blip_enc_enc_large.yaml"
        }

        assert model_type in paths, "Unknown model type {}".format(model_type)
        return paths[model_type]

    def forward_encoder_pre(self, samples):
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
        return {'image_embeds': self.encoder_pre(samples['vis_data'])}
    
    def forward_encoder_pst(self, samples, encoder_pre_out, **kwargs):
        pass

    @classmethod
    def build_model(cls, cfg=None, model_type="base"):
        if not cfg:
            # useful when building model without provided configuration file
            from utils.config import Config
            cfg = Config.build_model_config(config_path=cls.default_config_path(model_type)).model
        
        return cls._build_model_from_config(cfg)
    
    @classmethod
    def _build_model_from_config(cls, cfg):
        embed_dim = cfg.get("embed_dim", 256)

        # vision encoder
        encoder_vis = VisionTransformerEncoder.build_model(cfg) 
        vision_width = encoder_vis.vision_width
        if "vision_width" not in cfg:
            cfg.vision_width = vision_width

        # text encoder + multimodal encoder
        encoder_xmodal = XBertEncoder.build_model(cfg)
        model = cls(encoder_vis, encoder_xmodal, embed_dim=embed_dim)

        # load pre-trained weights
        pretrain_path = cfg.get("pretrained", None)
        if pretrain_path is not None:
            model, msg = cls.load_from_pretrained(model, url_or_filename=pretrain_path)
        
            assert len(msg.missing_keys) == 0, "Missing keys {}.".format(msg.missing_keys)
            assert len(msg.unexpected_keys) == 0, "Unexpected keys {}.".format(msg.unexpected_keys)

        return model 

    @staticmethod
    def load_from_pretrained(model, url_or_filename):
        raise NotImplementedError
        # # [TODO] move to utils for reuse
        # if is_url(url_or_filename):
        #     cached_file = download_cached_file(url_or_filename, check_hash=False, progress=True)
        #     checkpoint = torch.load(cached_file, map_location='cpu') 
        # elif os.path.isfile(url_or_filename):        
        #     checkpoint = torch.load(url_or_filename, map_location='cpu') 
        # else:
        #     raise RuntimeError('checkpoint url or path is invalid')

        # state_dict = checkpoint['model']
        
        # if "visual_encoder.pos_embed" in state_dict.keys():
        #     state_dict['visual_encoder.pos_embed'] = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'], model.encoder) 
        # elif "encoder.pos_embed" in state_dict.keys():
        #     state_dict['encoder.pos_embed'] = interpolate_pos_embed(state_dict['encoder.pos_embed'], model.encoder) 

        # pretrain_specific_keys = set([
        #     "temp", "image_queue", "text_queue", "queue_ptr", 
        #     "vision_proj.weight", "vision_proj.bias",
        #     "text_proj.weight", "text_proj.bias",
        #     "itm_head.weight", "itm_head.bias"]
        # )
        # # FIXME rename the keys in pre-trained state_dict() to avoid this hotfix.
        # new_state_dict = dict()
        # for key in state_dict.keys():
        #     if key in pretrain_specific_keys:
        #         continue
        #     elif "text_encoder" in key:
        #         continue
        #     elif "_m" in key:
        #         continue
        #     elif "visual_encoder" in key:
        #         new_key = key.replace("visual_encoder", "encoder")
        #     elif "text_decoder" in key:
        #         new_key = key.replace("text_decoder", "decoder")
        #     else:
        #         new_key = key
        #     new_state_dict[new_key] = state_dict[key]

        # # update old state_dict
        # state_dict = new_state_dict

        # # exclude incompatible keys
        # for key in model.state_dict().keys():
        #     if key in state_dict.keys():
        #         if state_dict[key].shape!=model.state_dict()[key].shape:
        #             del state_dict[key]
        
        # msg = model.load_state_dict(state_dict,strict=False)
        # print('load checkpoint from %s'%url_or_filename)  
        # return model, msg
    