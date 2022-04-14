from abc import abstractmethod

import torch.nn as nn


class BaseModel(nn.Module):
    """Base class for models."""

    def __init__(self):
        super().__init__()

    def forward_features(self, *args, **kwargs):
        """Similar to *forward* but only return features."""
        raise NotImplementedError

    def load_from_pretrained(self, url_or_filename):
        raise NotImplementedError

    @classmethod
    def build_from_cfg(cls, cfg):
        """
        A factory method to create instance from cfg.

        This is to ensure the definition of __init__() is not coupled to the cfg.
        Namely, even without cfg file, one should be able to recreate the instance.
        """
        raise NotImplementedError

    @property
    def device(self):
        return list(self.parameters())[0].device


class BaseEncoder(nn.Module):
    """
    Base class for primitive encoders, such as ViT, TimeSformer, etc.
    """

    def __init__(self):
        super().__init__()

    def forward(self, samples, **kwargs):
        raise NotImplementedError

    @property
    def device(self):
        return list(self.parameters())[0].device


class BaseDecoder(nn.Module):
    """Base class for decoders."""

    def __init__(self):
        super().__init__()

    def forward(self, samples, enc_out, **kwargs):
        """ """
        raise NotImplementedError


# class DualEncoderModel(BaseEncoder):
#     def __init__(self, vis_encoder, text_encoder):
#         super().__init__()

#         self.vis_encoder = vis_encoder
#         self.text_encoder = text_encoder

#     def forward(self, samples, **kwargs):
#         vis_input = samples.get("visual_input", None)
#         txt_input = samples.get("text_input", None)

#         assert not vis_input or not txt_input, "Visual and text inputs are both None."
#         vis_enc_out = self.vis_encoder(vis_input)
#         txt_enc_out = self.text_encoder(txt_input)

#         return {"vis_enc_out": vis_enc_out, "txt_enc_out": txt_enc_out}


class EncoderDecoderModel(BaseModel):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        assert isinstance(self.encoder, BaseEncoder)
        assert isinstance(self.decoder, BaseDecoder)

    @abstractmethod
    def forward_encoder(samples, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def forward_decoder(samples, encoder_out, **kwargs):
        raise NotImplementedError

    def forward(self, samples, **kwargs):
        encoder_out = self.forward_encoder(samples, **kwargs)
        decoder_out = self.forward_decoder(samples, encoder_out, **kwargs)
        return decoder_out

    def generate(self, samples, **kwargs):
        raise NotImplementedError


class EncoderEncoderModel(BaseEncoder):
    def __init__(self, encoder_pre, encoder_pst):
        super().__init__()

        self.encoder_pre = encoder_pre
        self.encoder_pst = encoder_pst

        assert isinstance(self.encoder_pre, BaseEncoder)
        assert isinstance(self.encoder_pst, BaseEncoder)

    @abstractmethod
    def forward_encoder_pre(samples, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def forward_decoder_pst(samples, encoder_pre_out, **kwargs):
        raise NotImplementedError

    def forward(self, samples, **kwargs):
        encoder_pre_out = self.forward_encoder_pre(samples, **kwargs)
        encoder_pst_out = self.forward_encoder_pst(samples, encoder_pre_out, **kwargs)

        return encoder_pst_out
