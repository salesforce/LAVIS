import torch
from einops import rearrange
from torch import nn
from typing import List, Optional, Tuple, Union
from .utils import extend_instance, stack_with_padding, num_params, getattr_recursive
from .cross_attn_lm import CrossAttentionMixin
from .helpers import DecoupledEmbedding, DecoupledLinear, VLMOutputWithPast
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import CLIPVisionModel
from transformers.models.siglip.modeling_siglip import SiglipVisionTransformer

from open_flamingo.train.any_res_data_utils import get_anyres_image_grid_shape, unpad_image


class VLM(nn.Module):
    """
    Generic vision-language model (VLM) class.
    A VLM consists of four components:
        1. A vision encoder that extracts features from pixels, e.g. CLIP
            input: (B, T_img, F, C, H, W)
            output: (B, T_img, F, v, d)
        2. A vision tokenizer that converts these features to visual token-like embeddings, e.g. Perceiver, or a linear projection head
            input: (B, T_img, F, v, d)
            output: (B, T_img, n, d)
        3. A fusion method that allows the language model to attend to these tokens, e.g. cross-attention, or placing the tokens directly in the language model's input sequence
        4. A language model
    """

    def __init__(
        self,
        vision_encoder: nn.Module,
        vision_tokenizer: nn.Module,
        lang_model: nn.Module,
        initial_tokenizer_len: int,
        pad_token_id: int,
        gradient_checkpointing: bool = False,
        base_img_size: Optional[int] = None,
    ):
        """
        Args:
            vision_encoder (nn.Module): e.g. CLIP
            vision_tokenizer (nn.Module): e.g. PerceiverResampler
            lang_model (nn.Module): e.g. MPT
            initial_tokenizer_len (int): size of the original tokenizer vocab
            pad_token_id (int): id of the pad token
            gradient_checkpointing (bool, optional): Whether to use gradient checkpointing. Defaults to False.
        """
        super().__init__()

        # save dimension information
        self.lang_embedding_dim = lang_model.get_input_embeddings().weight.shape[1]
        if hasattr(lang_model.config, "d_model"):
            self.lang_hidden_dim = lang_model.config.d_model  # mpt uses d_model
        else:
            self.lang_hidden_dim = lang_model.config.hidden_size
        self.vis_embedding_dim = vision_tokenizer.dim_media
        self.num_tokens_per_vis = vision_tokenizer.num_tokens_per_media

        # core components
        self.vision_encoder = vision_encoder
        self.vision_tokenizer = vision_tokenizer
        self.lang_model = lang_model

        if base_img_size is None:
            if isinstance(self.vision_encoder, CLIPVisionModel) or isinstance(self.vision_encoder, SiglipVisionTransformer):
                base_img_size = self.vision_encoder.config.image_size
            else:
                base_img_size = self.vision_encoder.image_size[0]
        self.base_img_size = base_img_size

        # lm embeddings
        self.pad_token_id = pad_token_id
        self.initial_tokenizer_len = initial_tokenizer_len
        input_embeds = DecoupledEmbedding(
            max_original_id=initial_tokenizer_len - 1,
            num_additional_embeddings=len(self.special_tokens),
            _weight=self.lang_model.get_input_embeddings().weight,
            pad_token_id=self.pad_token_id,
        )
        if hasattr(input_embeds, "additional_embedding"):
            input_embeds.additional_embedding.weight.data.normal_(
                mean=0.0,
                std=self.lang_model.config.initializer_range
                if hasattr(self.lang_model.config, "initializer_range")
                else 0.02,
            )
        self.lang_model.set_input_embeddings(input_embeds)

        out_embeds = DecoupledLinear(
            max_original_id=initial_tokenizer_len - 1,
            additional_out_features=len(self.special_tokens),
            _weight=self.lang_model.get_output_embeddings().weight,
            _bias=self.lang_model.get_output_embeddings().bias if hasattr(self.lang_model.get_output_embeddings(), "bias") else None,
        )
        if hasattr(out_embeds, "additional_fc"):
            out_embeds.additional_fc.weight.data.normal_(
                mean=0.0,
                std=self.lang_model.config.initializer_range
                if hasattr(self.lang_model.config, "initializer_range")
                else 0.02,
            )
        self.lang_model.set_output_embeddings(out_embeds)

        # gradient checkpointing
        self.vision_tokenizer._use_gradient_checkpointing = gradient_checkpointing

    def forward(
        self,
        vision_x: Optional[torch.Tensor],
        lang_x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[
            List[Union[torch.Tensor, Tuple[torch.Tensor]]]
        ] = None,
        past_media_locations: Optional[torch.Tensor] = None,
        past_vision_tokens: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = False,
        **kwargs,
    ):
        """
        Args:
            vision_x: Vision input
                shape (B, T_img, F, C, H, W) with F=1
                only F = 1 is supported (single-frame videos)
                if T_img > the number of media tokens in the corresponding input_ids (lang_x),
                only the first number of media tokens in lang_x are used
            lang_x: Language input ids, with media tokens denoting where
                visual media should be inserted.
                shape (B, T_txt)
            attention_mask: Attention mask. Defaults to None.
            labels: Labels. Defaults to None.
                shape (B, T_txt)
            past_key_values (Tuple[torch.Tensor]], optional): Past key value pairs for each of the T_txt previous tokens in the language model. Defaults to None.
                list of length = number of decoder layers in the LM
                exact implementation depends on LM, see Hugging Face docs
            past_media_locations (torch.Tensor, optional): boolean mask denoting which of the previous T_txt tokens were media tokens. Defaults to None.
                shape (B, T_txt)
            past_vision_tokens (torch.Tensor, optional): Previous vision tokens. Defaults to None.
            use_cache (Optional[bool], optional): Whether to use cache. Defaults to False.
                If True, includes key_values, media_locations, and vision_tokens in the output.
        """
        assert not (past_vision_tokens is None) ^ (
            past_media_locations is None
        ), "past_vision_tokens and past_media_locations must both be None or both be not None"

        # convert pixels to vision tokens
        if vision_x is not None:
            vision_features = self._encode_vision_x(vision_x=vision_x)
            vision_tokens = self.vision_tokenizer(vision_features)
        else:
            vision_tokens = None

        # fuse the vision and language tokens
        new_inputs = self._prepare_inputs_for_forward(
            vision_tokens=vision_tokens,
            lang_x=lang_x,
            attention_mask=attention_mask,
            labels=labels,
            past_key_values=past_key_values,
            past_media_locations=past_media_locations,
            padding_side="right",
            past_vision_tokens=past_vision_tokens,
        )
        output = self.lang_model(
            **new_inputs,
            use_cache=use_cache,
            past_key_values=past_key_values,
            **kwargs,
        )

        # postprocessing may be needed, e.g. to remove extra tokens from logits that were inserted into the language stream
        # or to add the past_vision_tokens and past_media_locations to the output
        output = self._postprocess_outputs_from_forward(
            output=output,
            lang_x=lang_x,
            vision_tokens=vision_tokens,
            use_cache=use_cache,
            past_vision_tokens=past_vision_tokens,
            past_media_locations=past_media_locations,
        )

        # postforward hooks
        self._post_forward_hook()
        return output

    def _encode_vision_x_anyres(self, samples, device):
        assert self.anyres_grids is not None
        image_raw = samples["image"] # list of patch list in of shape [1, N_patch, C, H, W]
        image_sizes = samples["image_size"]

        # Image_raw can be a list of list of patches, when a `samples` has multiple images.
        if isinstance(image_raw[0], list):
            images = [x.squeeze(0) for sample_img in image_raw for x in sample_img]
            image_sizes = [s for sample_sizes in image_sizes for s in sample_sizes]
        else:
            # assert isinstance(image_raw[0], torch.Tensor), f"Unkown image type: {image_raw[0]}"
            # concate list of patches into one big patch for any res encoding.
            images = [x.squeeze(0) for x in image_raw] # [N_patch, C, H, W]
        image = torch.cat(images, dim=0) # [\sum{B}{N_patch_i}, C, H, W]
        image = image.to(device)

        with torch.no_grad():
            if self.vision_encoder.__class__.__name__ == "TimmModel":
                image_embeds = self.vision_encoder.trunk.forward_features(image)
            elif self.vision_encoder.__class__.__name__ in ['CLIPVisionModel', 'SiglipVisionTransformer']:
                image_embeds = self.vision_encoder(image, interpolate_pos_encoding=True).last_hidden_state
            else:
                image_embeds = self.vision_encoder(image)[1]  # OpenCLIP returns tuples
        
        if self.vision_encoder.__class__.__name__ == "TimmModel":
            grid_size = self.vision_encoder.trunk.patch_embed.grid_size
        elif self.vision_encoder.__class__.__name__ in ['CLIPVisionModel', 'SiglipVisionTransformer']:
            grid_size_base = self.base_img_size // self.vision_encoder.config.patch_size
            grid_size = (grid_size_base, grid_size_base)
        else:
            grid_size = self.vision_encoder.grid_size
        height, width = grid_size
        
        if not image_embeds.shape[1] == height * width:
            assert image_embeds.shape[1] == height * width + 1 # For vision encoders that has [CLS] token.
            image_embeds = image_embeds[:, 1:, :] # Drop the cls token for each patch.
        n_vis_token_per_patch = image_embeds.shape[1]

        # Split encoded patches and merge patch features
        # 1. Get the raw sizes from samples, and split the image embeds [\sum_{B}(N_patch_i), N_tok(16*16), C]
        split_sizes = [image.shape[0] for image in images]
        image_embeds = torch.split(image_embeds, split_sizes, dim=0)
        # 2. For each image (consist of a list of patches), merge the patches spatially (of shape [C, n_patch_height, n_patch_width])
        new_image_embeds = []
        patch_attn_masks = []
        max_n_img_token = -1
        for idx, patch_embeds in enumerate(image_embeds):
            if patch_embeds.shape[0] > 1:
                # 3. Flatten the patch features and get [C, n_patch_height * (n_patch_width+1)]
                base_patch_embeds = patch_embeds[0] # TODO: prepend the CLS token for th base patch embeds (of the resized entire image).
                patch_embeds = patch_embeds[1:]
                
                assert height * width == base_patch_embeds.shape[0]
    
                num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[idx], 
                                                                                self.anyres_grids, 
                                                                                self.base_img_size) # Hardcoded grid_pinpoints.
                patch_embeds = patch_embeds.view(num_patch_height, num_patch_width, height, width, -1)

                patch_embeds = patch_embeds.permute(4, 0, 2, 1, 3).contiguous()
                patch_embeds = patch_embeds.flatten(1, 2).flatten(2, 3)
                patch_embeds, patch_attn_mask = unpad_image(patch_embeds, image_sizes[idx], self.anyres_patch_sampling)
                if hasattr(self, 'image_newline'):
                    patch_embeds = torch.cat((
                            patch_embeds,
                            self.image_newline[:, None, None].expand(*patch_embeds.shape[:-1], 1)
                        ), dim=-1)
                if self.anyres_patch_sampling:
                    patch_embeds = patch_embeds.view(-1, num_patch_height, num_patch_width, height*width)
                    patch_embeds = patch_embeds.flatten(1, 2).permute(1, 2, 0)
                    assert patch_attn_mask is not None
                    patch_attn_mask = patch_attn_mask.view(num_patch_height, num_patch_width, height*width)
                    patch_attn_mask = patch_attn_mask.flatten(0, 1)
                    patch_embeds = torch.cat((base_patch_embeds.unsqueeze(0), patch_embeds), dim=0)
                    patch_attn_mask = torch.cat((torch.ones(n_vis_token_per_patch, device=patch_embeds.device).unsqueeze(0), patch_attn_mask), dim=0)
                else:
                    patch_embeds = patch_embeds.flatten(1, 2).transpose(0, 1)
                    patch_embeds = torch.cat((base_patch_embeds, patch_embeds), dim=0)
            else:
                patch_embeds = patch_embeds[0].unsqueeze(0) if self.anyres_patch_sampling else patch_embeds[0]
                patch_attn_mask = torch.ones(n_vis_token_per_patch, device=patch_embeds.device).unsqueeze(0) if self.anyres_patch_sampling else None
                if hasattr(self, 'image_newline'):
                    patch_embeds = torch.cat((
                            patch_embeds,
                            self.image_newline[None]
                        ), dim=0)
            if not self.anyres_patch_sampling:
                max_n_img_token = max(patch_embeds.shape[0], max_n_img_token)

            new_image_embeds.append(patch_embeds)
            patch_attn_masks.append(patch_attn_mask)
        
        if self.anyres_patch_sampling:
            # Return individual patches for independent token downsampling.
            return new_image_embeds, patch_attn_masks

        # 4. Pad and concat the list of image_embeds [N_tok_i, C] together into a batch. Also modify the query attention mask.
        image_embeds = []
        image_atts = []
        for image_embed in new_image_embeds:
            n_img_token = image_embed.shape[0]
            img_attn = torch.ones((max_n_img_token), dtype=torch.long, device=image_embed.device)
            if n_img_token < max_n_img_token:
                padded_embed = torch.zeros((max_n_img_token, image_embed.shape[-1]), dtype=image_embed.dtype, device=image_embed.device)
                padded_embed[:n_img_token, :] = image_embed
                img_attn[n_img_token:] = 0 # Mask out the padded entries.
            else:
                padded_embed = image_embed
            image_embeds.append(padded_embed)
            image_atts.append(img_attn)
        image_embeds = torch.stack(image_embeds, dim=0) # Shape [B, N_tok_longest, C_dim]
        image_atts = torch.stack(image_atts, dim=0) # Shape [B, N_tok_longest, C_dim]
        # TODO: reshape image_embeds and image_atts to "b T F v d"
        image_embeds = image_embeds[:, None, None, :, :]
        # image_atts = image_atts[:, None, None, :, :]

        return image_embeds, image_atts

    def _encode_vision_x(self, vision_x: torch.Tensor):
        """
        Compute media tokens from vision input by passing it through vision encoder and conditioning language model.
        Args:
            vision_x: Vision input
                shape (B, T_img, F, C, H, W)
                Images in the same chunk are collated along T_img, and frames are collated along F
                Currently only F=1 is supported (single-frame videos)

        rearrange code based on https://github.com/dhansmair/flamingo-mini
        """
        assert vision_x.ndim == 6, "vision_x should be of shape (b, T_img, F, C, H, W)"
        b, T, F = vision_x.shape[:3]
        
        vision_x = rearrange(vision_x, "b T F c h w -> (b T F) c h w")
        with torch.no_grad():
            if self.vision_encoder.__class__.__name__ == "TimmModel":
                vision_x = self.vision_encoder.trunk.forward_features(vision_x)
            elif self.vision_encoder.__class__.__name__ in ['CLIPVisionModel', 'SiglipVisionTransformer']:
                vision_x = self.vision_encoder(vision_x).last_hidden_state
            else:
                vision_x = self.vision_encoder(vision_x)[1]  # OpenCLIP returns tuples
        vision_x = rearrange(vision_x, "(b T F) v d -> b T F v d", b=b, T=T, F=F)
        return vision_x

    def _concat_vision_cache(
        self, lang_x, vision_tokens, past_vision_tokens, past_media_locations, use_cache
    ):
        """
        Helper function to include the past vision tokens and past media locations in the output.
        """
        if use_cache:
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
                        lang_x == self.media_token_id,
                    ],
                    dim=1,
                )
            else:
                updated_vision_tokens = vision_tokens
                updated_media_locations = lang_x == self.media_token_id

        else:
            updated_vision_tokens = None
            updated_media_locations = None

        return updated_vision_tokens, updated_media_locations

    def generate(
        self,
        vision_x: torch.Tensor,
        lang_x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        past_key_values: Optional[
            List[Union[torch.Tensor, Tuple[torch.Tensor]]]
        ] = None,
        past_media_locations: Optional[torch.Tensor] = None,
        past_vision_tokens: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Generate text conditioned on vision and language inputs.
        Args:
            vision_x (torch.Tensor): Vision input
                shape (B, T_img, F, C, H, W)
                see documentation for forward
            lang_x (torch.Tensor): Language input
                shape (B, T_txt)
            attention_mask (torch.Tensor, optional): Attention mask. Defaults to None.
            **kwargs: see generate documentation in Hugging Face CausalLM models.
        Returns:
            torch.Tensor: lang_x with generated tokens appended to it
        """
        num_beams = kwargs.pop("num_beams", 1)

        # convert pixels to vision tokens
        if vision_x is not None:
            vision_features = self._encode_vision_x(vision_x=vision_x)
            vision_tokens = self.vision_tokenizer(vision_features)
        else:
            vision_tokens = None

        # fuse the vision and language tokens
        # for xattn, vision_x and media_location are repeat_interleaved s.t.
        # the total batch size is B * num_beams
        new_inputs = self._prepare_inputs_for_forward(
            vision_tokens=vision_tokens,
            lang_x=lang_x,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            past_media_locations=past_media_locations,
            past_vision_tokens=past_vision_tokens,
            padding_side="left",
            num_beams=num_beams,
        )
        output = self.lang_model.generate(
            **new_inputs,
            past_key_values=past_key_values,
            num_beams=num_beams,
            use_cache=True,
            **kwargs,
        )
        self._post_forward_hook()
        return output

    @property
    def num_trainable_params(self):
        """Print the number of trainable parameters"""
        return num_params(self, filter_to_trainable=True)

    def set_trainable(self):
        """
        Freeze appropriate parameters in the model.
        """
        raise NotImplementedError

    def group_params_by_weight_decay(self):
        """
        Return a tuple of (params to optimize w/ weight decay, params to optimize w/o weight decay)
        """
        params_with_wd, params_without_wd = [], []
        for n, p in self.named_parameters():
            if p.requires_grad:    
                if self._should_apply_weight_decay(n):
                    params_with_wd.append(p)
                else:
                    params_without_wd.append(p)
        return params_with_wd, params_without_wd

    def _should_apply_weight_decay(self, parameter_name):
        """
        Return whether weight decay should be applied to a parameter.
        """
        raise NotImplementedError

    @property
    def special_tokens(self):
        """
        Returns a dict mapping from the attribute name of a special token to its string format,
         e.g. "media_token": "<image>"
        """
        assert (
            "media_token" in self._special_tokens
        ), "VLMs need to request that the tokenizer add a media_token and call set_special_token_ids to set self.media_token_id"
        return self._special_tokens

    @property
    def special_token_ids(self):
        """
        Returns a list of the special token ids
        """
        return [getattr(self, f"{att_name}_id") for att_name in self.special_tokens]

    def set_special_token_ids(self, string_to_ids):
        """
        Args:
            string_to_ids (dict): mapping from token string to id
        """
        assert set(self.special_tokens.values()).issubset(set(string_to_ids.keys()))
        for att_name, token_str in self.special_tokens.items():
            token_id = string_to_ids[token_str]
            setattr(self, f"{att_name}_id", token_id)
            setattr(self.lang_model, f"{att_name}_id", token_id)

    def init_gradient_checkpointing(self):
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            checkpoint_wrapper,
            CheckpointWrapper,
            CheckpointImpl,
            apply_activation_checkpointing,
        )
        from functools import partial

        non_reentrant_wrapper = partial(
            checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )
        apply_activation_checkpointing(
            self,
            checkpoint_wrapper_fn=non_reentrant_wrapper,
            check_fn=lambda m: getattr(m, "_use_gradient_checkpointing", False)
            and not isinstance(m, CheckpointWrapper),
        )


class VLMWithCrossAttention(VLM):
    """
    VLM using cross-attention to fuse vision and language tokens.
    """

    def __init__(
        self,
        vision_encoder: nn.Module,
        vision_tokenizer: nn.Module,
        lang_model: nn.Module,
        initial_tokenizer_len: int,
        pad_token_id: int,
        gradient_checkpointing: bool = False,
        decoder_layers_attr_name: str = None,
        cross_attn_every_n_layers: int = None,
    ):
        extend_instance(lang_model, CrossAttentionMixin)
        super().__init__(
            vision_encoder=vision_encoder,
            vision_tokenizer=vision_tokenizer,
            lang_model=lang_model,
            initial_tokenizer_len=initial_tokenizer_len,
            pad_token_id=pad_token_id,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.lang_model.set_decoder_layers_attr_name(decoder_layers_attr_name)
        self.decoder_layers_attr_name = decoder_layers_attr_name
        self.lang_model.init_cross_attention_layers(
            lang_hidden_size=self.lang_hidden_dim,
            vis_hidden_size=self.vis_embedding_dim,
            cross_attn_every_n_layers=cross_attn_every_n_layers,
            gradient_checkpointing=gradient_checkpointing,
        )

    def _prepare_inputs_for_forward(
        self,
        vision_tokens: torch.Tensor,
        lang_x: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor = None,
        past_key_values=None,
        past_media_locations: torch.Tensor = None,
        past_vision_tokens: torch.Tensor = None,
        padding_side: str = "right",  # noop for cross-attention models
        num_beams: int = 1,
    ):
        """Each xattn layer needs to save the vision tokens and the locations of the media tokens in the language sequence"""
        self.lang_model._condition_media_before_forward(
            input_ids=lang_x,
            vision_tokens=vision_tokens,
            past_media_locations=past_media_locations,
            past_vision_tokens=past_vision_tokens,
            num_beams=num_beams,
        )
        if past_key_values is not None: 
            past_key_values = [
                (
                    k.repeat_interleave(num_beams, dim=0),
                    v.repeat_interleave(num_beams, dim=0)
                )
                for k, v in past_key_values
            ]

        return {
            "input_ids": lang_x,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _postprocess_outputs_from_forward(
        self,
        output: CausalLMOutputWithPast,
        lang_x: torch.Tensor,
        vision_tokens: torch.Tensor,
        past_vision_tokens: torch.Tensor,
        past_media_locations: torch.Tensor,
        use_cache: bool = False,
    ):
        """Include the past vision tokens and past media locations in the output"""
        updated_vision_tokens, updated_media_locations = self._concat_vision_cache(
            lang_x=lang_x,
            vision_tokens=vision_tokens,
            past_vision_tokens=past_vision_tokens,
            past_media_locations=past_media_locations,
            use_cache=use_cache,
        )
        output = VLMOutputWithPast(
            loss=output.loss,
            logits=output.logits,
            past_key_values=output.past_key_values,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
            past_media_locations=updated_media_locations,
            past_vision_tokens=updated_vision_tokens,
        )

        return output

    def _post_forward_hook(self):
        # clear the conditioned layers
        self.lang_model.clear_conditioned_layers()

    def get_fsdp_lambda_fn(self):
        """
        Returns the lambda function used to decide how to perform FSDP wrapping.
        """
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            CheckpointWrapper,
        )
        from .helpers import GatedCrossAttentionBlock

        decoder_block_class = getattr_recursive(
            self.lang_model, self.decoder_layers_attr_name
        )[0].__class__

        def lambda_fn(module: nn.Module):
            # we want FSDP(ckpt(module)), not ckpt(FSDP(module))
            if getattr(module, "_use_gradient_checkpointing", False) and not isinstance(
                module, CheckpointWrapper
            ):
                return False
            if module is self.vision_tokenizer:
                return True
            if isinstance(module, GatedCrossAttentionBlock):
                return True
            if isinstance(module, decoder_block_class):
                return True

        return lambda_fn

    @property
    def num_params_per_module(self):
        """Print the number of parameters per module in the model"""
        num_xattn_params = num_params(self.lang_model.gated_cross_attn_layers)
        return "\n".join(
            [
                f"Vision encoder: {num_params(self.vision_encoder):,} parameters",
                f"Vision tokenizer: {num_params(self.vision_tokenizer):,} parameters",
                f"Cross attention: {num_xattn_params:,} parameters",
                f"Language model: {num_params(self.lang_model) - num_xattn_params:,} parameters",
            ]
        )

    @property
    def num_trainable_params_per_module(self):
        """Print the number of trainable parameters per module in the model"""
        num_xattn_params = num_params(
            self.lang_model.gated_cross_attn_layers, filter_to_trainable=True
        )
        return "\n".join(
            [
                f"Vision encoder: {num_params(self.vision_encoder, filter_to_trainable=True):,} trainable parameters",
                f"Vision tokenizer: {num_params(self.vision_tokenizer, filter_to_trainable=True):,} trainable parameters",
                f"Cross attention: {num_xattn_params:,} trainable parameters",
                f"Language model: {num_params(self.lang_model, filter_to_trainable=True) - num_xattn_params:,} trainable parameters",
            ]
        )


class VLMWithLanguageStream(VLM):
    """
    VLM that fuses modalities by inserting vision tokens directly into the language stream.
    """

    def __init__(
        self,
        vision_encoder: nn.Module,
        vision_tokenizer: nn.Module,
        lang_model: nn.Module,
        initial_tokenizer_len: int,
        pad_token_id: int,
        decoder_layers_attr_name: str = None,
        gradient_checkpointing: bool = False,
        base_img_size: Optional[int] = None,
    ):
        super().__init__(
            vision_encoder=vision_encoder,
            vision_tokenizer=vision_tokenizer,
            lang_model=lang_model,
            initial_tokenizer_len=initial_tokenizer_len,
            pad_token_id=pad_token_id,
            base_img_size=base_img_size,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.decoder_layers_attr_name = decoder_layers_attr_name
        for block in getattr_recursive(self.lang_model, self.decoder_layers_attr_name):
            block._use_gradient_checkpointing = gradient_checkpointing

    def _prepare_inputs_for_forward(
        self,
        vision_tokens: torch.Tensor,
        lang_x: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor = None,
        past_key_values=None,
        vision_attention_mask: Optional[torch.Tensor] = None,
        past_media_locations: torch.Tensor = None,
        past_vision_tokens: torch.Tensor = None,
        padding_side: str = "left",
        num_beams: int = 1,
    ):
        """
        Insert the vision tokens directly into the language stream/
        This requires us to modify the input_ids, attention_mask, and labels.
        """
        if past_key_values is not None:
            past_len = past_key_values[0][0].shape[2]
            assert attention_mask.shape[1] == past_len + lang_x.shape[1], (
                "Attention_mask must be as long as the entire past len (including image tokens) and current input IDs. "
                + "Check that you've expanded the attention mask to account for past image tokens."
            )
        
        if vision_tokens is None:
            return {
                "input_ids": lang_x,
                "attention_mask": attention_mask,
                "labels": labels,
            }

        # get the language embeddings
        lang_embeds = self.lang_model.get_input_embeddings()(lang_x)

        # build up the multimodal embeddings
        B = lang_x.shape[0]
        has_labels = labels is not None
        multimodal_embeds = []
        multimodal_attention_mask = []
        multimodal_labels = [] if has_labels else None
        for i in range(B):
            # get index of <image> tokens in lang_x[i]
            image_token_idxs = torch.where(lang_x[i] == self.media_token_id)[0]

            if len(image_token_idxs) == 0:
                multimodal_embeds.append(lang_embeds[i].clone())
                multimodal_attention_mask.append(attention_mask[i].clone())
                if has_labels:
                    multimodal_labels.append(labels[i].clone())
                continue

            # since an image is represented by self.num_tokens_per_vis tokens, we need to offset the image_token_idxs

            # loop through the image_token_idxs and insert the vision tokens
            new_embed = lang_embeds[i].clone()
            new_attention_mask = (
                attention_mask[i].clone() if attention_mask is not None else None
            )
            if has_labels:
                new_label = labels[i].clone()

            for img_num in range(len(image_token_idxs)):
                img_idx = image_token_idxs[img_num]
                # Get vision token attention mask for padded llava-style any resolution image tokens.
                if self.image_aspect_ratio =='anyres':
                    num_vis_tokens = vision_tokens[i][img_num].shape[0]
                    if vision_attention_mask is not None:
                        vis_attention_mask = vision_attention_mask[i][img_num]
                    else:
                        vis_attention_mask = torch.ones(
                            num_vis_tokens, dtype=torch.long
                        ).to(attention_mask.device)
                else:
                    assert (
                        vision_tokens[i][img_num].shape[0] == self.num_tokens_per_vis
                    ), f"vision token number mismatch: image embedding ({vision_tokens[i][img_num].shape[0]}) \
                            vs. model.num_tokens_per_vis ({self.num_tokens_per_vis})"
                    # By default, vision tokens are not padded.
                    num_vis_tokens = self.num_tokens_per_vis
                    vis_attention_mask = torch.ones(
                        num_vis_tokens, dtype=torch.long
                    ).to(attention_mask.device)
                
                # Offset the rest of image tokens with current num_vis_tokens
                for j in range(img_num+1, len(image_token_idxs)):
                    image_token_idxs[j] += (num_vis_tokens - 1)

                new_embed = torch.cat(
                    (
                        new_embed[:img_idx],
                        vision_tokens[i][img_num],
                        new_embed[img_idx + 1 :],
                    ),
                    dim=0,
                )
                new_attention_mask = torch.cat(
                    (
                        new_attention_mask[:img_idx],
                        vis_attention_mask,
                        new_attention_mask[img_idx + 1 :],
                    ),
                    dim=0,
                )
                if has_labels:
                    new_label = torch.cat(
                        (
                            new_label[:img_idx],
                            torch.ones(num_vis_tokens, dtype=torch.long).to(
                                labels.device
                            )
                            * -100,
                            new_label[img_idx + 1 :],
                        ),
                        dim=0,
                    )
            multimodal_embeds.append(new_embed)
            multimodal_attention_mask.append(new_attention_mask)
            if has_labels:
                multimodal_labels.append(new_label)

        # stack
        multimodal_embeds = stack_with_padding(
            multimodal_embeds,
            padding_value=self.pad_token_id,
            padding_side=padding_side,
        )
        multimodal_attention_mask = stack_with_padding(
            multimodal_attention_mask,
            padding_value=0,
            padding_side=padding_side,
        )
        if has_labels:
            multimodal_labels = stack_with_padding(
                multimodal_labels,
                padding_value=-100,
                padding_side=padding_side,
            )

        return {
            "inputs_embeds": multimodal_embeds,
            "attention_mask": multimodal_attention_mask,
            "labels": multimodal_labels,
        }

    def _postprocess_outputs_from_forward(
        self,
        output: CausalLMOutputWithPast,
        lang_x: torch.Tensor,
        vision_tokens: torch.Tensor,
        past_vision_tokens: torch.Tensor,
        past_media_locations: torch.Tensor,
        use_cache: bool = False,
    ):
        # Include the past vision tokens and past media locations in the output
        updated_vision_tokens, updated_media_locations = self._concat_vision_cache(
            lang_x=lang_x,
            vision_tokens=vision_tokens,
            past_vision_tokens=past_vision_tokens,
            past_media_locations=past_media_locations,
            use_cache=use_cache,
        )

        # return logits that are the same shape as the original input_ids
        logits = output.logits
        batch_logits = []
        B, T_txt = lang_x.shape
        for i in range(B):
            sequence_logits = []
            logits_j = 0
            img_id = 0
            for j in range(T_txt):
                if lang_x[i, j] != self.media_token_id:
                    sequence_logits.append(logits[i, logits_j])
                    logits_j += 1
                else:
                    # append the logit for the first image token, then skip over the rest
                    # note: the model actually learns to predict <im_patch>, not <image>
                    sequence_logits.append(logits[i, logits_j])
                    # logits_j += self.num_tokens_per_vis
                    # Offset in account of dynamic num_vis_tokens.
                    logits_j += vision_tokens[i][img_id].shape[0]
                    img_id += 1
            sequence_logits = torch.stack(sequence_logits, dim=0)  # (B, vocab_size)
            batch_logits.append(sequence_logits)

        batch_logits = torch.stack(batch_logits, dim=0)  # (B, T_txt, vocab_size)
        # The final logits shape should be the same as the original input_ids shape
        assert batch_logits.shape[:2] == (B, T_txt)

        # assemble the output
        output = VLMOutputWithPast(
            loss=output.loss,
            logits=batch_logits,
            past_key_values=output.past_key_values,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
            past_media_locations=updated_media_locations,
            past_vision_tokens=updated_vision_tokens,
        )

        return output

    def _post_forward_hook(self):
        pass

    def get_fsdp_lambda_fn(self):
        """
        Returns the lambda function used to decide how to perform FSDP wrapping.
        """
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            CheckpointWrapper,
        )

        decoder_block_class = getattr_recursive(
            self.lang_model, self.decoder_layers_attr_name
        )[0].__class__

        def lambda_fn(module: nn.Module):
            if getattr(module, "_use_gradient_checkpointing", False) and not isinstance(
                module, CheckpointWrapper
            ):
                return False
            if module is self.vision_tokenizer:
                return True
            if isinstance(module, decoder_block_class):
                return True

        return lambda_fn
    
    def get_fsdp_wrapping_policy(self):
        """
        Returns the policy used to decide how to perform FSDP wrapping.
        """
        from torch.distributed.fsdp.wrap import _or_policy, _module_wrap_policy, transformer_auto_wrap_policy
        from open_clip.transformer import VisionTransformer, ResidualAttentionBlock
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer
        from transformers.models.phi.modeling_phi import PhiDecoderLayer
        # for Phi-3 hot fiix
        try:
            import importlib
            commit_hash = str(type(self.lang_model)).split('instruct.')[1].split('.modeling')[0]
            module_name = f"transformers_modules.microsoft.Phi-3-mini-128k-instruct.{commit_hash}.modeling_phi3"
            module = importlib.import_module(module_name)
            Phi3DecoderLayer = module.Phi3DecoderLayer
            import_phi3 = True
        except IndexError:
            import_phi3 = False
            
                
        # hard code the wrap module name
        # vision
        if isinstance(self.vision_encoder, SiglipVisionModel):
            from transformers import SiglipVisionModel
            vit_wrap_policy = functools.partial(_module_wrap_policy, module_classes={SiglipVisionModel})
            from transformers.models.siglip.modeling_siglip import SiglipEncoderLayer, SiglipVisionTransformer, SiglipVisionEmbeddings, SiglipMultiheadAttentionPoolingHead
            # import torch.nn.LayerNorm as LayerNorm
            transformer_layer_cls_vit = {SiglipEncoderLayer, SiglipVisionTransformer, SiglipVisionEmbeddings, SiglipMultiheadAttentionPoolingHead}
            vision_transformer_block_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls=transformer_layer_cls_vit)
            vision_wrap_policy = functools.partial(_or_policy, policies=[vit_wrap_policy, vision_transformer_block_policy])

        else:
            vit_wrap_policy = functools.partial(_module_wrap_policy, module_classes={VisionTransformer, TimmModel})
            # vit_wrap_policy = functools.partial(_module_wrap_policy, module_classes={VisionTransformer})
            # transformer_layer_cls_vit = {ResidualAttentionBlock}
            transformer_layer_cls_vit = {ResidualAttentionBlock, Block}
            # transformer_layer_cls_vit = {Block}
            vision_transformer_block_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls=transformer_layer_cls_vit)
            vision_wrap_policy = functools.partial(_or_policy, policies=[vit_wrap_policy, vision_transformer_block_policy])
        # llm
        transformer_layer_cls={LlamaDecoderLayer, PhiDecoderLayer}
        if import_phi3:
            transformer_layer_cls.add(Phi3DecoderLayer)
        llm_transformer_block_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls=transformer_layer_cls)
        # vision_tokenizer
        vis_tokenizer_policy = functools.partial(_module_wrap_policy, module_classes={LinearPatchProjection, PerceiverResampler})
        return functools.partial(
            _or_policy, 
            policies = [
                vision_wrap_policy, 
                llm_transformer_block_policy,
                vis_tokenizer_policy
            ])    

    @property
    def num_params_per_module(self):
        """Print the number of parameters per module in the model"""
        return "\n".join(
            [
                f"Vision encoder: {num_params(self.vision_encoder):,} parameters",
                f"Vision tokenizer: {num_params(self.vision_tokenizer):,} parameters",
                f"Language model: {num_params(self.lang_model):,} parameters",
            ]
        )

    @property
    def num_trainable_params_per_module(self):
        """Print the number of trainable parameters per module in the model"""
        return "\n".join(
            [
                f"Vision encoder: {num_params(self.vision_encoder, filter_to_trainable=True):,} trainable parameters",
                f"Vision tokenizer: {num_params(self.vision_tokenizer, filter_to_trainable=True):,} trainable parameters",
                f"Language model: {num_params(self.lang_model, filter_to_trainable=True):,} trainable parameters",
            ]
        )
