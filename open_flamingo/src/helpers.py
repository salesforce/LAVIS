"""
Based on: https://github.com/lucidrains/flamingo-pytorch
"""

import re
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from einops_exts import rearrange_many
from torch import einsum, nn
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Optional
from dataclasses import dataclass


@dataclass
class VLMOutputWithPast(CausalLMOutputWithPast):
    """
    VLMOutputWithPast is a wrapper around CausalLMOutputWithPast that adds the following attributes:
        past_media_locations: Optional[torch.Tensor] = None,
        past_vision_tokens: Optional[torch.Tensor] = None,
    """

    past_media_locations: Optional[torch.Tensor] = None
    past_vision_tokens: Optional[torch.Tensor] = None


def exists(val):
    return val is not None


def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


class VisionTokenizer(nn.Module):
    def __init__(self, dim_media, num_tokens_per_media):
        super().__init__()
        self.dim_media = dim_media
        self.num_tokens_per_media = num_tokens_per_media


class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents, vision_attn_masks=None):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, T, n1, D)
            latent (torch.Tensor): latent features
                shape (b, T, n2, D)
        """
        x = self.norm_media(x)
        latents = self.norm_latents(latents)

        h = self.heads

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2) # TODO: Change the shape of vision attention mask according to this.
        if vision_attn_masks is not None:
            vision_attn_masks = torch.cat((vision_attn_masks, 
                                            torch.ones((latents.shape[0], latents.shape[-2]), dtype=latents.dtype, device=latents.device)),
                                            dim=-1)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        q, k, v = rearrange_many((q, k, v), "b t n (h d) -> b h t n d", h=h)
        q = q * self.scale

        # attention
        sim = einsum("... i d, ... j d  -> ... i j", q, k)
        # Apply vision attention mask here.
        # Reference: https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html#torch.nn.functional.scaled_dot_product_attention
        if vision_attn_masks is not None:
            attn_bias = torch.zeros((q.size(0), 1, 1, q.size(-2), k.size(-2)), dtype=q.dtype, device=q.device)
            vision_attn_masks = repeat(vision_attn_masks, 'b n -> b 1 1 l n', l=q.size(-2))
            attn_bias.masked_fill_(vision_attn_masks.logical_not(), float("-inf"))
            sim += attn_bias

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)
        

        out = einsum("... i j, ... j d -> ... i d", attn, v)
        out = rearrange(out, "b h t n d -> b t n (h d)", h=h)
        return self.to_out(out)


class PerceiverResampler(VisionTokenizer):
    def __init__(
        self,
        *,
        dim,
        dim_inner=None,
        depth=6,
        dim_head=96,
        heads=16,
        num_latents=128,
        max_num_media=None,
        max_num_frames=None,
        ff_mult=4,
    ):
        """
        Perceiver module which takes in image features and outputs image tokens.
        Args:
            dim (int): dimension of the incoming image features
            dim_inner (int, optional): final dimension to project the incoming image features to;
                also the final dimension of the outputted features. If None, no projection is used, and dim_inner = dim.
            depth (int, optional): number of layers. Defaults to 6.
            dim_head (int, optional): dimension of each head. Defaults to 64.
            heads (int, optional): number of heads. Defaults to 8.
            num_latents (int, optional): number of latent tokens to use in the Perceiver;
                also corresponds to number of tokens per sequence to output. Defaults to 64.
            max_num_media (int, optional): maximum number of media per sequence to input into the Perceiver
                and keep positional embeddings for. If None, no positional embeddings are used.
            max_num_frames (int, optional): maximum number of frames to input into the Perceiver
                and keep positional embeddings for. If None, no positional embeddings are used.
            ff_mult (int, optional): dimension multiplier for the feedforward network. Defaults to 4.
        """
        if dim_inner is not None:
            projection = nn.Linear(dim, dim_inner)
        else:
            projection = None
            dim_inner = dim
        super().__init__(dim_media=dim, num_tokens_per_media=num_latents)
        self.projection = projection
        self.latents = nn.Parameter(torch.randn(num_latents, dim))

        # positional embeddings
        self.frame_embs = (
            nn.Parameter(torch.randn(max_num_frames, dim))
            if exists(max_num_frames)
            else None
        )
        self.media_time_embs = (
            nn.Parameter(torch.randn(max_num_media, 1, dim))
            if exists(max_num_media)
            else None
        )

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(
                            dim=dim, dim_head=dim_head, heads=heads
                        ),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

        self.norm = nn.LayerNorm(dim)

    def forward(self, x, vision_attn_masks):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, T, F, v, D)
            vision_attn_masks (torch.Tensor): attention masks for padded visiont tokens (i.e., x)
                shape (b, v)
        Returns:
            shape (b, T, n, D) where n is self.num_latents
        """
        b, T, F, v = x.shape[:4]

        # frame and media time embeddings
        if exists(self.frame_embs):
            frame_embs = repeat(self.frame_embs[:F], "F d -> b T F v d", b=b, T=T, v=v)
            x = x + frame_embs
        x = rearrange(
            x, "b T F v d -> b T (F v) d"
        )  # flatten the frame and spatial dimensions
        if exists(self.media_time_embs):
            x = x + self.media_time_embs[:T]

        # blocks
        latents = self.latents
        latents = repeat(latents, "n d -> b T n d", b=b, T=T)
        for attn, ff in self.layers:
            latents = attn(x, latents, vision_attn_masks) + latents
            latents = ff(latents) + latents
        
        if exists(self.projection):
            return self.projection(self.norm(latents)) 
        else:
            return self.norm(latents)

class LinearPatchProjection(VisionTokenizer):
    """Linear projection from patch features to image tokens."""

    def __init__(self, mm_projector_type, *, dim_visual, dim_out, num_patches):
        super().__init__(dim_media=dim_visual, num_tokens_per_media=num_patches)
        if mm_projector_type == 'linear':
            self.proj = nn.Linear(dim_visual, dim_out)
        else:
            mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', mm_projector_type)
            if mlp_gelu_match:
                mlp_depth = int(mlp_gelu_match.group(1))
                modules = [nn.Linear(dim_visual, dim_out)]
                for _ in range(1, mlp_depth):
                    modules.append(nn.GELU())
                    modules.append(nn.Linear(dim_out, dim_out))
                self.proj = nn.Sequential(*modules)
            else:
                raise ValueError(f'Unknown projector type: {mm_projector_type}')

    def forward(self, x):
        B = x.shape[0]
        x = rearrange(x, "b T F v d -> (b T) (F v) d")
        x = self.proj(x)
        return rearrange(x, "(b T) n d -> b T n d", b=B)
    
# gated cross attention
class MaskedCrossAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_visual,
        dim_head=64,
        heads=8,
        only_attend_immediate_media=True,
    ):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim_visual, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        # whether for text to only attend to immediate preceding image, or all previous images
        self.only_attend_immediate_media = only_attend_immediate_media

    def forward(self, x, media, media_locations=None):
        """
        Args:
            x (torch.Tensor): text features
                shape (B, T_txt, D_txt)
            media (torch.Tensor): image features
                shape (B, T_img, n, D_img) where n is the dim of the latents
            media_locations: boolean mask identifying the media tokens in x
                shape (B, T_txt_all)
                T_txt_all >= T_txt
                If T_txt_all > T_txt, then the last T_txt text_times are used
        """

        T_txt = x.shape[1]
        assert (
            T_txt <= media_locations.shape[1]
        ), "current text cannot be longer than conditioned media locations"

        _, T_img, n = media.shape[:3]
        h = self.heads

        x = self.norm(x)

        q = self.to_q(x)
        media = rearrange(media, "b t n d -> b (t n) d")

        k, v = self.to_kv(media).chunk(2, dim=-1)
        q, k, v = rearrange_many((q, k, v), "b n (h d) -> b h n d", h=h)

        q = q * self.scale

        sim = einsum("... i d, ... j d -> ... i j", q, k)

        if exists(media_locations):
            media_time = torch.arange(T_img, device=x.device) + 1

            # at each boolean of True, increment the time counter (relative to media time)
            text_time = media_locations.cumsum(dim=-1)[:, -T_txt:]

            # text time must equal media time if only attending to most immediate image
            # otherwise, as long as text time is greater than media time (if attending to all previous images / media)
            mask_op = torch.eq if self.only_attend_immediate_media else torch.ge

            text_to_media_mask = mask_op(
                rearrange(text_time, "b i -> b 1 i 1"),
                repeat(media_time, "j -> 1 1 1 (j n)", n=n),
            )
            sim = sim.masked_fill(~text_to_media_mask, -torch.finfo(sim.dtype).max)

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        if exists(media_locations) and self.only_attend_immediate_media:
            # any text without a preceding media needs to have attention zeroed out
            text_without_media_mask = text_time == 0
            text_without_media_mask = rearrange(
                text_without_media_mask, "b i -> b 1 i 1"
            )
            attn = attn.masked_fill(text_without_media_mask, 0.0)

        out = einsum("... i j, ... j d -> ... i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class GatedCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_visual,
        dim_head=64,
        heads=8,
        ff_mult=4,
        only_attend_immediate_media=True,
    ):
        super().__init__()
        self.attn = MaskedCrossAttention(
            dim=dim,
            dim_visual=dim_visual,
            dim_head=dim_head,
            heads=heads,
            only_attend_immediate_media=only_attend_immediate_media,
        )
        self.attn_gate = nn.Parameter(torch.tensor([0.0]))

        self.ff = FeedForward(dim, mult=ff_mult)
        self.ff_gate = nn.Parameter(torch.tensor([0.0]))

    def forward(
        self,
        x,
        media,
        media_locations=None,
    ):
        x = (
            self.attn(
                x,
                media,
                media_locations=media_locations,
            )
            * self.attn_gate.tanh()
            + x
        )
        x = self.ff(x) * self.ff_gate.tanh() + x

        return x


class QFormerWithProjection(VisionTokenizer):
    """
    Based on BLIP-2 (https://arxiv.org/pdf/2301.12597.pdf)
    In the BLIP-2 paper, Q-former is initialized with BERT-base weights,
    so dim_inner = 768, num_hidden_layers = 12, and intermediate_size = 3072
    """

    def __init__(
        self,
        dim_input,
        dim_out,
        dim_inner=768,
        num_hidden_layers=12,
        num_query_tokens=32,
    ):
        super().__init__(dim_media=dim_out, num_tokens_per_media=num_query_tokens)
        # initialize the qformer
        from transformers import Blip2QFormerModel, Blip2QFormerConfig

        self.qformer = Blip2QFormerModel(
            Blip2QFormerConfig(
                encoder_hidden_size=dim_input,
                hidden_size=dim_inner,
                num_hidden_layers=num_hidden_layers,
            )
        )
        self.query_tokens = nn.Parameter(
            torch.zeros(1, num_query_tokens, dim_inner)
        )
        self.proj = nn.Linear(dim_inner, dim_out)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): image features
                shape (B, T, F, v, D)
        Returns:
            shape (B, T, n, D) where n is num_query_tokens
        """
        # HF class expects three dimensional input
        B, T = x.shape[:2]
        x = rearrange(x, "b T F v d -> (b T) (F v) d")

        # get the outputs
        image_attention_mask = torch.ones(
            x.size()[:-1], dtype=torch.long, device=x.device
        )
        query_tokens = self.query_tokens.expand(x.shape[0], -1, -1)
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=x,
            encoder_attention_mask=image_attention_mask,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        query_output = query_outputs[0]
        query_output = self.proj(query_output)

        # reshape
        query_output = rearrange(query_output, "(b T) n d -> b T n d", b=B)
        return query_output


# Both DecoupledEmbedding and DecoupledLinear are taken from https://github.com/huggingface/transformers/blob/v4.32.1/src/transformers/models/idefics/modeling_idefics.py and renamed for clarity
class DecoupledEmbedding(nn.Embedding):
    # Derived from https://pytorch.org/docs/stable/_modules/torch/nn/modules/sparse.html#Embedding
    """
    Implements a decoupling of parameters to allow freezing (or not) a subset of the embeddings. In practise, the
    regular `weight` can be trained or frozen (i.e. `partially_freeze=True`), and if `num_additional_embeddings` > 0,
    then it will create `num_additional_embeddings` additional parameters that are always trained. If
    `num_additional_embeddings=0`, then the module defaults back to the regular behavior of `nn.Embedding`.
    """

    def __init__(
        self,
        max_original_id: int,
        num_additional_embeddings: int = 0,
        _weight: torch.Tensor = None,
        num_original_embeddings: int = None,
        embedding_dim: int = None,
        partially_freeze=True,
        device=None,
        dtype=None,
        pad_token_id=None,
    ) -> None:
        """
        Args:
            max_original_id (`int`):
                The largest token id that should be embedded using the regular embedding (regular `weight`).
                This is usually len(tokenizer) - 1 before additional tokens are added.
                Note that this may not equal self.weight.shape[0]
            num_additional_embeddings (`int`):
                Number of additional tokens to initialize an Embedding matrix for (`additional_weight`).
            _weight (`torch.Tensor`, *optional*, defaults to `None`): The regular weight tensor.
                If provided, this sets the `num_original_embeddings` and `embedding_dim` parameters.
            num_original_embeddings (`int`):
                self.weight.shape[0]
            embedding_dim (`int`):
                The size of each embedding vector
            partially_freeze: (`bool`, *optional*, defaults to `True`):
                If `True`, the regular `weight` will be frozen. `additional_weight` is never frozen.
            padding_idx (`int`, *optional*):
                The padding index (needs to be less than num_embeddings)

        Note: there are a lot of other parameters to initialize a standard `nn.Embedding` such as `padding_idx`,
        `max_norm` or `norm_type`. We are not supporting these.
        """
        # validate args
        if pad_token_id is not None and pad_token_id > max_original_id:
            raise ValueError(
                f"pad_token_id must be <= max_original_id. Got {pad_token_id} and {max_original_id}."
                + "If the original tokenizer does not have a pad_token_id, use pad_token_id=None."
            )
        if _weight is not None:
            assert (num_original_embeddings is None) or (
                _weight.shape[0] == num_original_embeddings
            ), f"num_original_embeddings={num_original_embeddings} but _weight.shape[0]={_weight.shape[0]}"
            assert (embedding_dim is None) or (
                _weight.shape[1] == embedding_dim
            ), f"embedding_dim={embedding_dim} but _weight.shape[1]={_weight.shape[1]}"
            num_original_embeddings = _weight.shape[0]
            embedding_dim = _weight.shape[1]
        else:
            assert (
                num_original_embeddings is not None
            ), "num_original_embeddings must be provided if _weight is not provided"
            assert (
                embedding_dim is not None
            ), "embedding_dim must be provided if _weight is not provided"

        super().__init__(
            num_embeddings=num_original_embeddings,
            embedding_dim=embedding_dim,
            device=device,
            dtype=dtype,
            padding_idx=pad_token_id,
            _weight=_weight,
        )
        self.max_original_id = max_original_id
        self.padding_idx = pad_token_id
        self.num_additional_embeddings = num_additional_embeddings
        if self.num_additional_embeddings > 0:
            self.additional_embedding = nn.Embedding(
                num_embeddings=self.num_additional_embeddings,
                embedding_dim=embedding_dim,
                device=device,
                dtype=dtype,
            )
        self.set_requires_grad(
            require_regular_grad=not partially_freeze, require_additional_grad=True
        )

    def set_requires_grad(self, require_regular_grad, require_additional_grad):
        """
        Helper function to separately set the requires_grad flag for the regular weight and the additional weight.
        """
        self.weight.requires_grad_(require_regular_grad)
        self.additional_embedding.requires_grad_(require_additional_grad)

    def forward(self, input_ids):
        """
        we have 2 embeddings, with different indices - one pretrained self.weight and another
        self.additional_embedding.weight that is being trained.

        in order to make a lookup of the input ids, we:
        1. find out the indices of the entries belonging to the 2nd embedding
        2. extract those values while subtracting the size of the first embedding (num_embeddings), since the 2nd
        embedding starts from 0 and not num_embeddings
        3. perform the 2nd embedding lookup
        4. now we handle the 1st embedding, we overwrite indices belonging to the 2nd embedding with a padding index
        5. perform the 1st embedding lookup
        6. now we overwrite the values in the 1st embedding lookup with the values of the 2nd embedding lookup

        note: for the 1st embedding lookup we could have looked up only the low indices and not do the padding, but
        then we have to create a new tensor and populate it with 2 tensors that are spread out across various indices -
        i.e. not a simple concat - I haven't benchmarked the complex case if it's any faster, given that seqlens are
        usually relatively short it's probably not faster or if faster not by much - but might be a good idea to
        measure.

        """
        if self.num_additional_embeddings == 0:
            return F.embedding(input_ids, self.weight)

        # Clone so that we don't modify the original input_ids later on
        input_ids = input_ids.clone()
        additional_vocab_indices = torch.where(input_ids > self.max_original_id)
        input_ids_additional_vocab = input_ids[additional_vocab_indices]
        additional_embeddings = self.additional_embedding(
            input_ids_additional_vocab - self.max_original_id - 1
        )

        # for successful lookup replace input_ids with 0, the results of these will be discarded anyway
        input_ids[additional_vocab_indices] = 0
        full_vector = F.embedding(input_ids, self.weight)

        # overwrite the records with high indices
        full_vector[additional_vocab_indices] = additional_embeddings

        return full_vector

    def extra_repr(self) -> str:
        return "num_original_embeddings={}, num_additional_embeddings={}, embedding_dim={}, partially_freeze={}".format(
            self.max_original_id + 1,
            self.num_additional_embeddings,
            self.embedding_dim,
            (not self.weight.requires_grad),
        )


class DecoupledLinear(nn.Linear):
    # Derived from https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear
    """
    Implements a decoupling of parameters to allow freezing (or not) a subset of the parameters. In practise, the
    regular `weight` can be trained or frozen (i.e. `partially_freeze=True`), and if `additional_out_features` > 0,
    then it will create `additional_out_features * in_features` additional parameters that are always trained. If
    `additional_out_features=0`, then the module defaults back to the regular behavior of `nn.Linear`.
    """

    def __init__(
        self,
        max_original_id: int,
        additional_out_features: int = 0,
        _weight: torch.Tensor = None,
        _bias: torch.Tensor = None,
        in_features: int = None,
        original_out_features: int = None,
        bias: bool = True,
        partially_freeze: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        """
        Args:
            max_original_id (`int`): The largest token id that should be extracted from the regular weight.
                This is usually len(tokenizer) - 1 before additional tokens are added.
                Note that this may not equal original_out_features - 1
            _weight: torch.Tensor, *optional*, defaults to `None`. The regular weight tensor.
                If provided, this sets the `in_features` and `original_out_features` parameters.
            _bias: torch.Tensor, *optional*, defaults to `None`. The regular bias tensor.
            in_features: int. Input hidden size.
            original_out_features: int. Original out_features of the language model's get_output_embeddings() function.
            additional_out_features: int. Number of additional trainable dimensions.
            bias: bool. Whether to include a bias term.
            partially_freeze: bool, *optional*, defaults to `True`): If `True`, the regular `weight` will be frozen.
        """
        # argument validation
        if _weight is not None:
            assert (_weight.shape[0] == original_out_features) or (
                original_out_features is None
            ), f"original_out_features={original_out_features} but _weight.shape[0]={_weight.shape[0]}"
            assert (_weight.shape[1] == in_features) or (
                in_features is None
            ), f"in_features={in_features} but _weight.shape[1]={_weight.shape[1]}"
            in_features = _weight.shape[1]
            original_out_features = _weight.shape[0]
        else:
            assert (
                in_features is not None
            ), "in_features must be provided if _weight is not provided"
            assert (
                original_out_features is not None
            ), "original_out_features must be provided if _weight is not provided"

        if _bias is not None:
            assert bias is True, "bias must be True if _bias is provided"

        # initialize original linear
        super().__init__(
            in_features, 
            original_out_features,
            bias, 
            device, 
            dtype)
        
        # set weight and bias manually
        if _weight is not None:
            self.weight = nn.Parameter(_weight)
        if _bias is not None:
            self.bias = nn.Parameter(_bias)
            
        self.in_features = in_features
        self.original_out_features = original_out_features
        self.max_original_id = max_original_id

        # initialize additional linear
        self.additional_out_features = additional_out_features
        self.has_bias = bias
        if additional_out_features > 0:
            self.additional_fc = nn.Linear(
                in_features=in_features,
                out_features=additional_out_features,
                bias=self.has_bias,
                device=device,
                dtype=dtype,
            )
        self.set_requires_grad(
            require_regular_grad=not partially_freeze, require_additional_grad=True
        )

    def set_requires_grad(self, require_regular_grad, require_additional_grad):
        """
        Helper function to separately set the requires_grad flag for the regular weight and the additional weight.
        """
        self.weight.requires_grad_(require_regular_grad)
        if self.has_bias:
            self.bias.requires_grad_(require_regular_grad)
        self.additional_fc.requires_grad_(require_additional_grad)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = F.linear(input, self.weight, self.bias)
        output = output[..., : self.max_original_id + 1]

        if self.additional_out_features > 0:
            additional_features = F.linear(
                input, self.additional_fc.weight, self.additional_fc.bias
            )
            output = torch.cat((output, additional_features), -1)
        return output

    def extra_repr(self) -> str:
        """Overwriting `nn.Linear.extra_repr` to include new parameters."""
        return "in_features={}, out_features={}, additional_out_features={}, bias={}, partially_freeze={}".format(
            self.in_features,
            self.max_original_id + 1,
            self.additional_out_features,
            self.bias is not None,
            (not self.weight.requires_grad or not self.bias.requires_grad),
        )
