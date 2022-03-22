import torch.nn as nn


class BaseModel(nn.Module):
    """Base class for models."""

    def __init__(self):
        super().__init__()
    
    def extract_features(self, *args, **kwargs):
        """Similar to *forward* but only return features."""
        raise NotImplementedError

    def load_from_pretrained(self, url_or_filename):
        raise NotImplementedError
    
    @classmethod
    def build_model(cls, cfg):
        """
        A factory method to create instance from cfg.

        This is to ensure the definition of __init__() is not coupled to the cfg.
        Namely, even without cfg file, one should be able to recreate the instance.
        """
        raise NotImplementedError

class BaseEncoder(nn.Module):
    """
    Base class for primitive encoders, such as ViT, TimeSformer, etc.
    """

    def __init__(self):
        super().__init__()
    
    def forward(self, samples, **kwargs):
        raise NotImplementedError


class BaseDecoder(nn.Module):
    """Base class for decoders."""

    def __init__(self):
        super().__init__()

    def forward(self, samples, enc_out, **kwargs):
        """
        """
        raise NotImplementedError


class DualEncoderModel(BaseEncoder):
    def __init__(self, vis_encoder, text_encoder):
        super().__init__()

        self.vis_encoder = vis_encoder
        self.text_encoder = text_encoder
    
    def forward(self, samples, **kwargs):
        vis_input = samples.get("visual_input", None)
        txt_input = samples.get("text_input", None)

        assert not vis_input or not txt_input, "Visual and text inputs are both None."
        vis_enc_out = self.vis_encoder(vis_input)
        txt_enc_out = self.vis_enc_out(txt_input)

        return {"vis_enc_out": vis_enc_out, "txt_enc_out": txt_enc_out}


class EncoderDecoderModel(BaseModel):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        assert isinstance(self.encoder, BaseEncoder)
        assert isinstance(self.decoder, BaseDecoder)
    
    def forward(self, samples, **kwargs):
        encoder_out = self.forward_encoder(samples, **kwargs)
        decoder_out = self.forward_decoder(samples, encoder_out, **kwargs)

        return decoder_out

    def generate(self, samples, **kwargs):
        raise NotImplementedError

class CascadeEncoderModel(BaseModel):
    def __init__(self, encoders):
        super().__init__()