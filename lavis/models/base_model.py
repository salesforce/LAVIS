from omegaconf import OmegaConf

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
    def build(cls, cfg=None, model_type="base"):
        if not cfg:
            # useful when building model without provided configuration file
            from common.config import Config

            default_config = OmegaConf.load(cls.default_config_path(model_type))
            cfg = Config.build_model_config(default_config).model

        return cls._build_from_cfg(cfg)

    @classmethod
    def build_default_model(cls, model_type="base"):
        return cls.build(cfg=None, model_type=model_type)

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

    def extract_features(self, samples, **kwargs):
        raise NotImplementedError

    @property
    def device(self):
        return list(self.parameters())[0].device


# class BaseDecoder(nn.Module):
#     """Base class for decoders."""

#     def __init__(self):
#         super().__init__()

#     def forward(self, samples, enc_out, **kwargs):
#         """ """
#         raise NotImplementedError
