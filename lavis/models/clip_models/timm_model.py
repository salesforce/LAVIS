"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

 Based on https://github.com/mlfoundations/open_clip
"""

""" timm model adapter
Wraps timm (https://github.com/rwightman/pytorch-image-models) models for use as a vision tower in CLIP model.
"""
import math
import warnings
from collections import OrderedDict
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import nn as nn

try:
    import timm
    from timm.models.layers import Mlp, to_2tuple

    # from timm.models.layers.attention_pool2d import RotAttentionPool2d
    # from timm.models.layers.attention_pool2d import (
    #     AttentionPool2d as AbsAttentionPool2d,
    # )

except ImportError as e:
    timm = None

from lavis.models.clip_models.utils import freeze_batch_norm_2d


class TimmModel(nn.Module):
    """timm model adapter
    # FIXME this adapter is a work in progress, may change in ways that break weight compat
    """

    def __init__(
        self,
        model_name,
        embed_dim,
        image_size=224,
        pool="avg",
        proj="linear",
        drop=0.0,
        pretrained=False,
    ):
        super().__init__()
        if timm is None:
            raise RuntimeError("Please `pip install timm` to use timm models.")

        self.image_size = to_2tuple(image_size)
        self.trunk = timm.create_model(model_name, pretrained=pretrained)
        feat_size = self.trunk.default_cfg.get("pool_size", None)
        feature_ndim = 1 if not feat_size else 2
        if pool in ("abs_attn", "rot_attn"):
            assert feature_ndim == 2
            # if attn pooling used, remove both classifier and default pool
            self.trunk.reset_classifier(0, global_pool="")
        else:
            # reset global pool if pool config set, otherwise leave as network default
            reset_kwargs = dict(global_pool=pool) if pool else {}
            self.trunk.reset_classifier(0, **reset_kwargs)
        prev_chs = self.trunk.num_features

        head_layers = OrderedDict()
        if pool == "abs_attn":
            head_layers["pool"] = AttentionPool2d(
                prev_chs, feat_size=feat_size, out_features=embed_dim
            )
            prev_chs = embed_dim
        elif pool == "rot_attn":
            head_layers["pool"] = RotAttentionPool2d(prev_chs, out_features=embed_dim)
            prev_chs = embed_dim
        else:
            assert proj, "projection layer needed if non-attention pooling is used."

        # NOTE attention pool ends with a projection layer, so proj should usually be set to '' if such pooling is used
        if proj == "linear":
            head_layers["drop"] = nn.Dropout(drop)
            head_layers["proj"] = nn.Linear(prev_chs, embed_dim)
        elif proj == "mlp":
            head_layers["mlp"] = Mlp(prev_chs, 2 * embed_dim, embed_dim, drop=drop)

        self.head = nn.Sequential(head_layers)

    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        """lock modules
        Args:
            unlocked_groups (int): leave last n layer groups unlocked (default: 0)
        """
        if not unlocked_groups:
            # lock full model
            for param in self.trunk.parameters():
                param.requires_grad = False
            if freeze_bn_stats:
                freeze_batch_norm_2d(self.trunk)
        else:
            # NOTE: partial freeze requires latest timm (master) branch and is subject to change
            try:
                # FIXME import here until API stable and in an official release
                from timm.models.helpers import group_modules, group_parameters
            except ImportError:
                raise RuntimeError(
                    "Please install latest timm `pip install git+https://github.com/rwightman/pytorch-image-models`"
                )
            matcher = self.trunk.group_matcher()
            gparams = group_parameters(self.trunk, matcher)
            max_layer_id = max(gparams.keys())
            max_layer_id = max_layer_id - unlocked_groups
            for group_idx in range(max_layer_id + 1):
                group = gparams[group_idx]
                for param in group:
                    self.trunk.get_parameter(param).requires_grad = False
            if freeze_bn_stats:
                gmodules = group_modules(self.trunk, matcher, reverse=True)
                gmodules = {k for k, v in gmodules.items() if v <= max_layer_id}
                freeze_batch_norm_2d(self.trunk, gmodules)

    def forward(self, x):
        x = self.trunk(x)
        x = self.head(x)
        return x


class RotAttentionPool2d(nn.Module):
    """Attention based 2D feature pooling w/ rotary (relative) pos embedding.
    This is a multi-head attention based replacement for (spatial) average pooling in NN architectures.
    Adapted from the AttentionPool2d in CLIP w/ rotary embedding instead of learned embed.
    https://github.com/openai/CLIP/blob/3b473b0e682c091a9e53623eebc1ca1657385717/clip/model.py
    NOTE: While this impl does not require a fixed feature size, performance at differeing resolutions from
    train varies widely and falls off dramatically. I'm not sure if there is a way around this... -RW
    """

    def __init__(
        self,
        in_features: int,
        out_features: int = None,
        embed_dim: int = None,
        num_heads: int = 4,
        qkv_bias: bool = True,
    ):
        super().__init__()
        embed_dim = embed_dim or in_features
        out_features = out_features or in_features
        self.qkv = nn.Linear(in_features, embed_dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(embed_dim, out_features)
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.pos_embed = RotaryEmbedding(self.head_dim)

        trunc_normal_(self.qkv.weight, std=in_features**-0.5)
        nn.init.zeros_(self.qkv.bias)

    def forward(self, x):
        B, _, H, W = x.shape
        N = H * W
        x = x.reshape(B, -1, N).permute(0, 2, 1)

        x = torch.cat([x.mean(1, keepdim=True), x], dim=1)

        x = (
            self.qkv(x)
            .reshape(B, N + 1, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = x[0], x[1], x[2]

        qc, q = q[:, :, :1], q[:, :, 1:]
        sin_emb, cos_emb = self.pos_embed.get_embed((H, W))
        q = apply_rot_embed(q, sin_emb, cos_emb)
        q = torch.cat([qc, q], dim=2)

        kc, k = k[:, :, :1], k[:, :, 1:]
        k = apply_rot_embed(k, sin_emb, cos_emb)
        k = torch.cat([kc, k], dim=2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N + 1, -1)
        x = self.proj(x)
        return x[:, 0]


class AttentionPool2d(nn.Module):
    """Attention based 2D feature pooling w/ learned (absolute) pos embedding.
    This is a multi-head attention based replacement for (spatial) average pooling in NN architectures.
    It was based on impl in CLIP by OpenAI
    https://github.com/openai/CLIP/blob/3b473b0e682c091a9e53623eebc1ca1657385717/clip/model.py
    NOTE: This requires feature size upon construction and well prevent adaptive sizing of the network.
    """

    def __init__(
        self,
        in_features: int,
        feat_size: Union[int, Tuple[int, int]],
        out_features: int = None,
        embed_dim: int = None,
        num_heads: int = 4,
        qkv_bias: bool = True,
    ):
        super().__init__()

        embed_dim = embed_dim or in_features
        out_features = out_features or in_features
        assert embed_dim % num_heads == 0
        self.feat_size = to_2tuple(feat_size)
        self.qkv = nn.Linear(in_features, embed_dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(embed_dim, out_features)
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5

        spatial_dim = self.feat_size[0] * self.feat_size[1]
        self.pos_embed = nn.Parameter(torch.zeros(spatial_dim + 1, in_features))
        trunc_normal_(self.pos_embed, std=in_features**-0.5)
        trunc_normal_(self.qkv.weight, std=in_features**-0.5)
        nn.init.zeros_(self.qkv.bias)

    def forward(self, x):
        B, _, H, W = x.shape
        N = H * W
        assert self.feat_size[0] == H
        assert self.feat_size[1] == W
        x = x.reshape(B, -1, N).permute(0, 2, 1)
        x = torch.cat([x.mean(1, keepdim=True), x], dim=1)
        x = x + self.pos_embed.unsqueeze(0).to(x.dtype)

        x = (
            self.qkv(x)
            .reshape(B, N + 1, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = x[0], x[1], x[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N + 1, -1)
        x = self.proj(x)
        return x[:, 0]


def pixel_freq_bands(
    num_bands: int,
    max_freq: float = 224.0,
    linear_bands: bool = True,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
):
    if linear_bands:
        bands = torch.linspace(1.0, max_freq / 2, num_bands, dtype=dtype, device=device)
    else:
        bands = 2 ** torch.linspace(
            0, math.log(max_freq, 2) - 1, num_bands, dtype=dtype, device=device
        )
    return bands * torch.pi


def inv_freq_bands(
    num_bands: int,
    temperature: float = 100000.0,
    step: int = 2,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    inv_freq = 1.0 / (
        temperature
        ** (torch.arange(0, num_bands, step, dtype=dtype, device=device) / num_bands)
    )
    return inv_freq


def build_sincos2d_pos_embed(
    feat_shape: List[int],
    dim: int = 64,
    temperature: float = 10000.0,
    reverse_coord: bool = False,
    interleave_sin_cos: bool = False,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Args:
        feat_shape:
        dim:
        temperature:
        reverse_coord: stack grid order W, H instead of H, W
        interleave_sin_cos: sin, cos, sin, cos stack instead of sin, sin, cos, cos
        dtype:
        device:
    Returns:
    """
    assert (
        dim % 4 == 0
    ), "Embed dimension must be divisible by 4 for sin-cos 2D position embedding"
    pos_dim = dim // 4
    bands = inv_freq_bands(
        pos_dim, temperature=temperature, step=1, dtype=dtype, device=device
    )

    if reverse_coord:
        feat_shape = feat_shape[::-1]  # stack W, H instead of H, W
    grid = (
        torch.stack(
            torch.meshgrid(
                [torch.arange(s, device=device, dtype=dtype) for s in feat_shape]
            )
        )
        .flatten(1)
        .transpose(0, 1)
    )
    pos2 = grid.unsqueeze(-1) * bands.unsqueeze(0)
    # FIXME add support for unflattened spatial dim?

    stack_dim = (
        2 if interleave_sin_cos else 1
    )  # stack sin, cos, sin, cos  instead of sin sin cos cos
    pos_emb = torch.stack([torch.sin(pos2), torch.cos(pos2)], dim=stack_dim).flatten(1)
    return pos_emb


def build_fourier_pos_embed(
    feat_shape: List[int],
    bands: Optional[torch.Tensor] = None,
    num_bands: int = 64,
    max_res: int = 224,
    linear_bands: bool = False,
    include_grid: bool = False,
    concat_out: bool = True,
    in_pixels: bool = True,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> List[torch.Tensor]:
    if bands is None:
        if in_pixels:
            bands = pixel_freq_bands(
                num_bands,
                float(max_res),
                linear_bands=linear_bands,
                dtype=dtype,
                device=device,
            )
        else:
            bands = inv_freq_bands(num_bands, step=1, dtype=dtype, device=device)
    else:
        if device is None:
            device = bands.device
        if dtype is None:
            dtype = bands.dtype

    if in_pixels:
        grid = torch.stack(
            torch.meshgrid(
                [
                    torch.linspace(-1.0, 1.0, steps=s, device=device, dtype=dtype)
                    for s in feat_shape
                ]
            ),
            dim=-1,
        )
    else:
        grid = torch.stack(
            torch.meshgrid(
                [torch.arange(s, device=device, dtype=dtype) for s in feat_shape]
            ),
            dim=-1,
        )
    grid = grid.unsqueeze(-1)
    pos = grid * bands

    pos_sin, pos_cos = pos.sin(), pos.cos()
    out = (grid, pos_sin, pos_cos) if include_grid else (pos_sin, pos_cos)
    # FIXME torchscript doesn't like multiple return types, probably need to always cat?
    if concat_out:
        out = torch.cat(out, dim=-1)
    return out


class FourierEmbed(nn.Module):
    def __init__(
        self,
        max_res: int = 224,
        num_bands: int = 64,
        concat_grid=True,
        keep_spatial=False,
    ):
        super().__init__()
        self.max_res = max_res
        self.num_bands = num_bands
        self.concat_grid = concat_grid
        self.keep_spatial = keep_spatial
        self.register_buffer(
            "bands", pixel_freq_bands(max_res, num_bands), persistent=False
        )

    def forward(self, x):
        B, C = x.shape[:2]
        feat_shape = x.shape[2:]
        emb = build_fourier_pos_embed(
            feat_shape,
            self.bands,
            include_grid=self.concat_grid,
            dtype=x.dtype,
            device=x.device,
        )
        emb = emb.transpose(-1, -2).flatten(len(feat_shape))
        batch_expand = (B,) + (-1,) * (x.ndim - 1)

        # FIXME support nD
        if self.keep_spatial:
            x = torch.cat(
                [x, emb.unsqueeze(0).expand(batch_expand).permute(0, 3, 1, 2)], dim=1
            )
        else:
            x = torch.cat(
                [x.permute(0, 2, 3, 1), emb.unsqueeze(0).expand(batch_expand)], dim=-1
            )
            x = x.reshape(B, feat_shape.numel(), -1)

        return x


def rot(x):
    return torch.stack([-x[..., 1::2], x[..., ::2]], -1).reshape(x.shape)


def apply_rot_embed(x: torch.Tensor, sin_emb, cos_emb):
    return x * cos_emb + rot(x) * sin_emb


def apply_rot_embed_list(x: List[torch.Tensor], sin_emb, cos_emb):
    if isinstance(x, torch.Tensor):
        x = [x]
    return [t * cos_emb + rot(t) * sin_emb for t in x]


def apply_rot_embed_split(x: torch.Tensor, emb):
    split = emb.shape[-1] // 2
    return x * emb[:, :split] + rot(x) * emb[:, split:]


def build_rotary_pos_embed(
    feat_shape: List[int],
    bands: Optional[torch.Tensor] = None,
    dim: int = 64,
    max_freq: float = 224,
    linear_bands: bool = False,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
):
    """
    NOTE: shape arg should include spatial dim only
    """
    feat_shape = torch.Size(feat_shape)

    sin_emb, cos_emb = build_fourier_pos_embed(
        feat_shape,
        bands=bands,
        num_bands=dim // 4,
        max_res=max_freq,
        linear_bands=linear_bands,
        concat_out=False,
        device=device,
        dtype=dtype,
    )
    N = feat_shape.numel()
    sin_emb = sin_emb.reshape(N, -1).repeat_interleave(2, -1)
    cos_emb = cos_emb.reshape(N, -1).repeat_interleave(2, -1)
    return sin_emb, cos_emb


class RotaryEmbedding(nn.Module):
    """Rotary position embedding
    NOTE: This is my initial attempt at impl rotary embedding for spatial use, it has not
    been well tested, and will likely change. It will be moved to its own file.
    The following impl/resources were referenced for this impl:
    * https://github.com/lucidrains/vit-pytorch/blob/6f3a5fcf0bca1c5ec33a35ef48d97213709df4ba/vit_pytorch/rvt.py
    * https://blog.eleuther.ai/rotary-embeddings/
    """

    def __init__(self, dim, max_res=224, linear_bands: bool = False):
        super().__init__()
        self.dim = dim
        self.register_buffer(
            "bands",
            pixel_freq_bands(dim // 4, max_res, linear_bands=linear_bands),
            persistent=False,
        )

    def get_embed(self, shape: List[int]):
        return build_rotary_pos_embed(shape, self.bands)

    def forward(self, x):
        # assuming channel-first tensor where spatial dim are >= 2
        sin_emb, cos_emb = self.get_embed(x.shape[2:])
        return apply_rot_embed(x, sin_emb, cos_emb)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)
