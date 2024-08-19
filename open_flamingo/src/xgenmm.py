import torch
from einops import rearrange
from torch import nn
from typing import List, Optional, Tuple, Union

import os

from .helpers import PerceiverResampler
from .vlm import VLMWithLanguageStream

class XGenMMPerceiver(VLMWithLanguageStream):
    def __init__(
        self,
        vision_encoder: nn.Module,
        lang_model: nn.Module,
        vis_feature_dim: int,
        initial_tokenizer_len: int,
        pad_token_id: int,
        decoder_layers_attr_name: str = None,
        gradient_checkpointing: bool = False,
        base_img_size: Optional[int] = None,
        image_aspect_ratio: str = 'anyres',
        anyres_patch_sampling: bool = True, 
        num_vision_tokens: int = 128,
    ):
        """
        Args:
            vision_encoder (nn.Module): HF CLIPModel
            lang_encoder (nn.Module): HF causal language model
            vis_feature_dim (int): final dimension of the visual features outputted by the vision_encoder
            initial_tokenizer_len (int): size of the tokenizer vocab
            padding_token_id (int): id of the padding token. None if no padding token; then a padding token
                will be inserted into self.special_tokens, which factory.py fills after creating new tokens
            decoder_layers_attr_name (str, optional): name of the decoder layers attribute. Defaults to None.
            gradient_checkpointing (bool, optional): whether to use gradient checkpointing. Defaults to False.
        """
        self._special_tokens = {
            "media_token": "<image>",
            "image_placeholder_token": "<image placeholder>",
            "end_of_trunk_token": "<|endofchunk|>",
        }
        lang_embedding_dim = lang_model.get_input_embeddings().weight.shape[1]
        super().__init__(
            vision_encoder=vision_encoder,
            vision_tokenizer=PerceiverResampler(
                dim=vis_feature_dim, dim_inner=lang_embedding_dim,
                num_latents=num_vision_tokens,
            ),
            lang_model=lang_model,
            initial_tokenizer_len=initial_tokenizer_len,
            gradient_checkpointing=gradient_checkpointing,
            base_img_size=base_img_size,
            decoder_layers_attr_name=decoder_layers_attr_name,
            pad_token_id=pad_token_id,
        )
        self.image_aspect_ratio = image_aspect_ratio
        self.anyres_patch_sampling = anyres_patch_sampling
        self.anyres_grids = None

    def set_trainable(self):
        """
        Unfreeze everything except the vision_encoder
        """
        self.requires_grad_(True)
        self.vision_encoder.requires_grad_(False)

    def _should_apply_weight_decay(self, parameter_name):
        """
        Kosmos applies 0.01 weight deacy to everything
        """
        return True
    
    def forward(
        self,
        vision_x: Optional[torch.Tensor],
        lang_x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        image_size: Optional[Tuple] = None,
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
        vision_attention_mask = None
        if vision_x is not None:
            if self.image_aspect_ratio == 'anyres':
                input_dict = dict(image=vision_x, image_size=image_size)
                vision_features, vision_attn_masks = self._encode_vision_x_anyres(input_dict, lang_x.device)
            else:
                vision_features = self._encode_vision_x(vision_x=vision_x)
                vision_attn_masks = None
            if self.anyres_patch_sampling:
                split_sizes = [feature.shape[0] for feature in vision_features]
                # Nested splits for multi-image samples.
                if isinstance(vision_x[0], list):
                    nt_images = [len(images) for images in vision_x]
                    split_split_sizes = []
                    img_id = 0
                    for nt in nt_images:
                        split_split_sizes.append(split_sizes[img_id:img_id+nt])
                        img_id += nt
                else:
                    nt_images = [1] * len(vision_x)
                    split_split_sizes = split_sizes
                vision_features = torch.cat(vision_features, dim=0)
                vision_features = vision_features[:, None, None, :, :] # Expand dimensions.
                vision_attn_masks = torch.cat(vision_attn_masks, dim=0)
            vision_tokens = self.vision_tokenizer(vision_features, vision_attn_masks)
            
            # Post-processing: Split the batches into groups of patches and concatenate them together.
            if self.anyres_patch_sampling:
                # assert isinstance(vision_x, list)
                if isinstance(vision_x[0], list):
                    vision_token_groups = torch.split(vision_tokens, list(sum(nt_img) for nt_img in split_split_sizes), dim=0)
                    vision_tokens = []
                    
                    for sample_id, patch_vis_tokens in enumerate(vision_token_groups):
                        patch_vis_token_groups =  torch.split(patch_vis_tokens, split_split_sizes[sample_id], dim=0) # [Np*nt, 1, v, d] -> [[Np_t, 1, v, d], ...]
                        flatten_vision_tokens = []
                        # padded_attn_masks = []
                        for image_vis_token in patch_vis_token_groups:
                            image_vis_token = image_vis_token.flatten(0, 2) # [Np, 1, v, d] -> [Np*v, d]
                            flatten_vision_tokens.append(image_vis_token)
                        vision_tokens_i = flatten_vision_tokens
                        vision_tokens.append(vision_tokens_i)
                else:
                    vision_token_groups = torch.split(vision_tokens, split_sizes, dim=0)
                    vision_tokens = []
                    for patch_vis_tokens in vision_token_groups:
                        patch_vis_tokens = patch_vis_tokens.flatten(0, 2) # [Np, 1, v, d] -> [Np*v, d]
                        vision_tokens.append(patch_vis_tokens.unsqueeze(0)) # Add the nt dimension.
        else:
            vision_tokens = None

        # fuse the vision and language tokens
        new_inputs = self._prepare_inputs_for_forward(
            vision_tokens=vision_tokens,
            lang_x=lang_x,
            attention_mask=attention_mask,
            vision_attention_mask=vision_attention_mask,
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

        # postforward hooks
        self._post_forward_hook()
        return output
    
    def generate(
        self,
        vision_x: torch.Tensor,
        lang_x: torch.Tensor,
        image_size: Optional[Tuple] = None,
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
        vision_attention_mask = None
        if vision_x is not None:
            if self.image_aspect_ratio == 'anyres':
                input_dict = dict(image=vision_x, image_size=image_size)
                vision_features, vision_attn_masks = self._encode_vision_x_anyres(input_dict, lang_x.device)
            else:
                vision_features = self._encode_vision_x(vision_x=vision_x)
                vision_attn_masks = None
            if self.anyres_patch_sampling:
                split_sizes = [feature.shape[0] for feature in vision_features]
                # Nested splits for multi-image samples.
                if isinstance(vision_x[0], list):
                    nt_images = [len(images) for images in vision_x]
                    split_split_sizes = []
                    img_id = 0
                    for nt in nt_images:
                        split_split_sizes.append(split_sizes[img_id:img_id+nt])
                        img_id += nt
                else:
                    nt_images = [1] * len(vision_x)
                    split_split_sizes = split_sizes
                vision_features = torch.cat(vision_features, dim=0)
                vision_features = vision_features[:, None, None, :, :] # Expand dimensions.
                vision_attn_masks = torch.cat(vision_attn_masks, dim=0)
            vision_tokens = self.vision_tokenizer(vision_features, vision_attn_masks)
            
            # Post-processing: Split the batches into groups of patches and concatenate them together.
            if self.anyres_patch_sampling:
                assert isinstance(vision_x, list)
                if isinstance(vision_x[0], list):
                    vision_token_groups = torch.split(vision_tokens, list(sum(nt_img) for nt_img in split_split_sizes), dim=0)
                    vision_tokens = []
                    
                    for sample_id, patch_vis_tokens in enumerate(vision_token_groups):
                        # Pad the image tokens within a sample.
                        patch_vis_token_groups =  torch.split(patch_vis_tokens, split_split_sizes[sample_id], dim=0) # [Np*nt, 1, v, d] -> [[Np_t, 1, v, d], ...]
                        flatten_vision_tokens = []
                        for image_vis_token in patch_vis_token_groups:
                            image_vis_token = image_vis_token.flatten(0, 2) # [Np, 1, v, d] -> [Np*v, d]
                            flatten_vision_tokens.append(image_vis_token)
                        vision_tokens_i = flatten_vision_tokens
                        vision_tokens.append(vision_tokens_i)
                else:
                    # Padding. FIXME: padding here might not be necessary?
                    vision_token_groups = torch.split(vision_tokens, split_sizes, dim=0)
                    # Padding.
                    vision_tokens = []
                    for patch_vis_tokens in vision_token_groups:
                        patch_vis_tokens = patch_vis_tokens.flatten(0, 2) # [Np, 1, v, d] -> [Np*v, d]
                        vision_tokens.append(patch_vis_tokens.unsqueeze(0)) # Add the nt dimension.
        else:
            vision_tokens = None

        # fuse the vision and language tokens
        new_inputs = self._prepare_inputs_for_forward(
            vision_tokens=vision_tokens,
            lang_x=lang_x,
            attention_mask=attention_mask,
            vision_attention_mask=vision_attention_mask,
            past_key_values=past_key_values,
            past_media_locations=past_media_locations,
            past_vision_tokens=past_vision_tokens,
            padding_side="left",
            num_beams=num_beams,
        )
        if past_key_values is not None:
            output = self.lang_model.generate(
                **new_inputs,
                past_key_values=past_key_values,
                num_beams=num_beams,
                use_cache=True,
                **kwargs,
            )
        else:
            output = self.lang_model.generate(
                **new_inputs,
                num_beams=num_beams,
                use_cache=True,
                **kwargs,
            )
        self._post_forward_hook()
        return output

