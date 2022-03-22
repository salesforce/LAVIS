import os
from urllib.parse import urlparse

import torch
from common.registry import registry
from timm.models.hub import download_cached_file

from models.base_model import EncoderDecoderModel
from models.blip import init_tokenizer
from models.med import BertLMHeadDecoder
from models.vit import VisionTransformerEncoder, interpolate_pos_embed


# [TODO] move to utils for reuse
def is_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")


@registry.register_model("blip_enc_dec")
class BlipEncoderDecoder(EncoderDecoderModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @classmethod
    def default_config_path(cls, vit_type="base"):
        paths = {
            "base": "configs/models/blip_enc_dec_base.yaml",
            "large": "configs/models/blip_enc_dec_large.yaml"
        }

        assert vit_type in paths, "Unknown ViT type {}".format(vit_type)
        return paths[vit_type]

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
    
    def forward_decoder(self, **kwargs):
        raise NotImplementedError

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
        decoder_out = self.decoder.generate(
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
    def build_model(cls, cfg):
        # vision encoder
        encoder = VisionTransformerEncoder(cfg) 
        if "vision_width" not in cfg:
            cfg.vision_width = encoder.vision_width
        # text encoder + multimodal decoder
        decoder = BertLMHeadDecoder(cfg)

        model = cls(encoder, decoder)

        # tokenizer and prompt
        model.tokenizer = init_tokenizer()
        model.prompt = cfg.get("prompt", None)
        model.prompt_length = len(model.tokenizer(model.prompt).input_ids) - 1
        
        # load pre-trained weights
        pretrain_path = cfg.get("pretrained", None)
        if pretrain_path is not None:
            model, msg = cls.load_from_pretrained(model, url_or_filename=pretrain_path)
        
        assert len(msg.missing_keys) == 0, "Missing keys {}.".format(msg.missing_keys)
        assert len(msg.unexpected_keys) == 0, "Unexpected keys {}.".format(msg.unexpected_keys)
        return model 


    @staticmethod
    def load_from_pretrained(model, url_or_filename):
        # [TODO] move to utils for reuse
        if is_url(url_or_filename):
            cached_file = download_cached_file(url_or_filename, check_hash=False, progress=True)
            checkpoint = torch.load(cached_file, map_location='cpu') 
        elif os.path.isfile(url_or_filename):        
            checkpoint = torch.load(url_or_filename, map_location='cpu') 
        else:
            raise RuntimeError('checkpoint url or path is invalid')

        state_dict = checkpoint['model']
        
        state_dict['encoder.vit.pos_embed'] = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'], model.encoder.vit) 
        del state_dict['visual_encoder.pos_embed']
        # state_dict['visual_encoder.pos_embed'] = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder) 

        # FIXME rename the keys in pre-trained state_dict() to avoid this hotfix.
        new_state_dict = dict()
        for key in state_dict.keys():
            if "visual_encoder" in key:
                new_key = key.replace("visual_encoder", "encoder.vit")
            elif "text_decoder" in key:
                new_key = key.replace("text_decoder", "decoder.text_decoder")
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
    