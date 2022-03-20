import torch.nn as nn


class BaseModel(nn.Module):
    """Base class for models."""

    def __init__(self):
        super().__init__()
    
    def extract_features(self, *args, **kwargs):
        """Similar to *forward* but only return features."""
        return self(*args, **kwargs)