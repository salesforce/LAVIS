import torch
from lavis.common.registry import registry

from lavis.models.blip_models.blip import BlipBase
from lavis.models.blip_models.blip_outputs import (
    BlipOutput,
    BlipIntermediateOutput,
)
from lavis.models.med import XBertLMHeadDecoder
from lavis.models.vit import VisionTransformerEncoder


@registry.register_model("blip_caption")
class BlipCaption(BlipBase):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "base": "configs/models/blip_caption_base.yaml",
        "base_coco": "configs/models/blip_caption_base_coco.yaml",
        "large": "configs/models/blip_caption_large.yaml",
        "large_coco": "configs/models/blip_caption_large_coco.yaml",
    }

    def __init__(self, image_encoder, text_decoder, prompt=None, max_txt_len=40):
        super().__init__()

        self.tokenizer = self.init_tokenizer()

        self.visual_encoder = image_encoder
        self.text_decoder = text_decoder

        self.prompt = prompt
        self.prompt_length = len(self.tokenizer(self.prompt).input_ids) - 1

        self.max_txt_len = max_txt_len

    def forward_encoder(self, samples):
        image_embeds = self.visual_encoder.forward_features(samples["image"])
        return image_embeds

    def forward_decoder(self, samples, image_embeds):
        # prepare inputs for forwarding decoder
        raw_text = samples["text_input"]
        text = self.tokenizer(
            raw_text,
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(self.device)
        text.input_ids[:, 0] = self.tokenizer.bos_token_id

        # prepare targets for forwarding decoder
        decoder_targets = text.input_ids.masked_fill(
            text.input_ids == self.tokenizer.pad_token_id, -100
        )
        decoder_targets[:, : self.prompt_length] = -100

        # forward decoder
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            self.device
        )
        decoder_output = self.text_decoder(
            input_ids=text.input_ids,
            attention_mask=text.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            labels=decoder_targets,
            return_dict=True,
        )

        return decoder_output, decoder_targets

    def forward(self, samples):
        image_embeds = self.forward_encoder(samples)
        decoder_output, decoder_targets = self.forward_decoder(samples, image_embeds)

        # return decoder_out
        return BlipOutput(
            loss=decoder_output.loss,
            loss_lm=decoder_output.loss,
            intermediate_output=BlipIntermediateOutput(
                image_embeds=image_embeds,
                decoder_output=decoder_output,
                decoder_labels=decoder_targets,
            ),
        )

    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=3,
        max_length=30,
        min_length=10,
        top_p=0.9,
        repetition_penalty=1.0,
        num_captions=1,
    ):
        # prepare inputs for decoder generation.
        encoder_out = self.forward_encoder(samples)
        image_embeds = torch.repeat_interleave(encoder_out, num_captions, 0)

        prompt = [self.prompt] * image_embeds.size(0)
        prompt = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt.input_ids[:, 0] = self.tokenizer.bos_token_id
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
            repetition_penalty=repetition_penalty,
        )

        outputs = self.tokenizer.batch_decode(decoder_out, skip_special_tokens=True)
        captions = [output[len(self.prompt) :] for output in outputs]

        return captions

    @classmethod
    def from_config(cls, cfg):
        # vision encoder
        image_encoder = VisionTransformerEncoder.from_config(cfg)
        # text encoder + multimodal decoder
        text_decoder = XBertLMHeadDecoder.from_config(cfg)

        prompt = cfg.get("prompt", None)
        max_txt_len = cfg.get("max_txt_len", 40)

        model = cls(image_encoder, text_decoder, prompt=prompt, max_txt_len=max_txt_len)

        # load pre-trained weights
        pretrain_path = cfg.get("pretrained", None)
        if pretrain_path is not None:
            msg = model.load_from_pretrained(url_or_filename=pretrain_path)

        return model
