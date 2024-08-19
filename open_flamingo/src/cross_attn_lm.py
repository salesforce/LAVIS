import torch.nn as nn
import torch
from .helpers import GatedCrossAttentionBlock
from .utils import getattr_recursive, setattr_recursive


class DecoderLayerWithCrossAttention(nn.Module):
    """
    DecoderLayerWithCrossAttention is a wrapper around the GatedCrossAttentionBlock and DecoderLayer.
    """

    def __init__(
        self, gated_cross_attn_layer, decoder_layer, gradient_checkpointing=False
    ):
        super().__init__()
        self.gated_cross_attn_layer = gated_cross_attn_layer
        self.decoder_layer = decoder_layer
        self.vis_x = None
        self.media_locations = None
        if self.gated_cross_attn_layer is not None:
            self.gated_cross_attn_layer._use_gradient_checkpointing = (
                gradient_checkpointing
            )
        self.decoder_layer._use_gradient_checkpointing = gradient_checkpointing

    def is_conditioned(self) -> bool:
        """Check whether the layer is conditioned."""
        return self.vis_x is not None and self.media_locations is not None

    # Used this great idea from this implementation of Flamingo (https://github.com/dhansmair/flamingo-mini/)
    def condition_vis_x(self, vis_x):
        self.vis_x = vis_x

    def condition_media_locations(self, media_locations):
        self.media_locations = media_locations

    def forward(
        self,
        lang_x,
        attention_mask=None,
        **decoder_layer_kwargs,
    ):
        # Cross attention
        contains_media = (self.media_locations == 1).any()
        if contains_media and self.gated_cross_attn_layer is not None:
            if self.vis_x is None:
                raise ValueError("vis_x must be conditioned before forward pass")

            if self.media_locations is None:
                raise ValueError(
                    "media_locations must be conditioned before forward pass"
                )

            lang_x = self.gated_cross_attn_layer(
                lang_x,
                self.vis_x,
                media_locations=self.media_locations,
            )

        # Normal decoder layer
        lang_x = self.decoder_layer(
            lang_x, attention_mask=attention_mask, **decoder_layer_kwargs
        )
        return lang_x


class CrossAttentionMixin(nn.Module):
    """
    Mixin to add cross-attention layers to a language model.
    """

    def set_decoder_layers_attr_name(self, decoder_layers_attr_name):
        self.decoder_layers_attr_name = decoder_layers_attr_name

    def _get_decoder_layers(self):
        return getattr_recursive(self, self.decoder_layers_attr_name)

    def _set_decoder_layers(self, value):
        setattr_recursive(self, self.decoder_layers_attr_name, value)

    def init_cross_attention_layers(
        self,
        lang_hidden_size,
        vis_hidden_size,
        cross_attn_every_n_layers,
        gradient_checkpointing,
    ):
        """
        Add gated cross attn layers to the decoder.
        """
        old_decoder_blocks = self._get_decoder_layers()
        self.decoder_block_class = old_decoder_blocks[0].__class__
        self.gated_cross_attn_layers = nn.ModuleList(
            [
                GatedCrossAttentionBlock(
                    dim=lang_hidden_size, dim_visual=vis_hidden_size
                )
                if (layer_idx + 1) % cross_attn_every_n_layers == 0
                else None
                for layer_idx, _ in enumerate(old_decoder_blocks)
            ]
        )
        self._set_decoder_layers(
            nn.ModuleList(
                [
                    DecoderLayerWithCrossAttention(
                        gated_cross_attn_layer, decoder_layer, gradient_checkpointing
                    )
                    for gated_cross_attn_layer, decoder_layer in zip(
                        self.gated_cross_attn_layers, old_decoder_blocks
                    )
                ]
            )
        )
        self.initialized_cross_attention = True

    def _condition_media_before_forward(
        self,
        input_ids: torch.Tensor,
        vision_tokens: torch.Tensor = None,
        past_media_locations: torch.Tensor = None,
        past_vision_tokens: torch.Tensor = None,
        num_beams: int = 1,
    ):
        """Each xattn layer needs to save the vision tokens and the locations of the media tokens in the language sequence"""
        assert (
            self.initialized_cross_attention
        ), "Cross attention layers have not been initialized. "

        # concat with past
        if past_media_locations is not None and past_vision_tokens is not None:
            if vision_tokens is not None:
                updated_vision_tokens = torch.cat(
                    [
                        past_vision_tokens,
                        vision_tokens,
                    ],
                    dim=1,
                )
            else:
                updated_vision_tokens = past_vision_tokens
            updated_media_locations = torch.cat(
                [
                    past_media_locations,
                    input_ids == self.media_token_id,
                ],
                dim=1,
            )
        else:
            updated_vision_tokens = vision_tokens
            updated_media_locations = input_ids == self.media_token_id

        # repeat the vision tokens and media locations for each beam
        updated_vision_tokens = updated_vision_tokens.repeat_interleave(
            num_beams, dim=0
        )
        updated_media_locations = updated_media_locations.repeat_interleave(
            num_beams, dim=0
        )

        # condition
        for layer in self._get_decoder_layers():
            layer.condition_vis_x(updated_vision_tokens)
            layer.condition_media_locations(updated_media_locations)

    def is_conditioned(self) -> bool:
        """Check whether all decoder layers are already conditioned."""
        return all(l.is_conditioned() for l in self._get_decoder_layers())

    def clear_conditioned_layers(self):
        for layer in self._get_decoder_layers():
            layer.condition_vis_x(None)
            layer.condition_media_locations(None)