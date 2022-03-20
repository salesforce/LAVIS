import torch.nn as nn

class BaseDecoder(nn.Module):
    """Base class for decoders."""

    def __init__(self):
        super().__init__()