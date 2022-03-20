import torch.nn as nn


class BaseEncoderModel(nn.Module):
    """Base class for models."""

    def __init__(self):
        super().__init__()
    
    def forward(self, visual_input, text_input, **kwargs):
        """
        """
        raise NotImplementedError

    def extract_features(self, *args, **kwargs):
        """Similar to *forward* but only return features."""
        return self(*args, **kwargs)