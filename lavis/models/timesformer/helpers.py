"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

 Based on https://github.com/facebookresearch/TimeSformer
"""

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright 2020 Ross Wightman
# Modified model creation / weight loading / state_dict helpers

import logging, warnings
import os
import math
from collections import OrderedDict

import torch
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F


def load_state_dict(checkpoint_path, use_ema=False):
    if checkpoint_path and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict_key = "state_dict"
        if isinstance(checkpoint, dict):
            if use_ema and "state_dict_ema" in checkpoint:
                state_dict_key = "state_dict_ema"
        if state_dict_key and state_dict_key in checkpoint:
            new_state_dict = OrderedDict()
            for k, v in checkpoint[state_dict_key].items():
                # strip `module.` prefix
                name = k[7:] if k.startswith("module") else k
                new_state_dict[name] = v
            state_dict = new_state_dict
        elif "model_state" in checkpoint:
            state_dict_key = "model_state"
            new_state_dict = OrderedDict()
            for k, v in checkpoint[state_dict_key].items():
                # strip `model.` prefix
                name = k[6:] if k.startswith("model") else k
                new_state_dict[name] = v
            state_dict = new_state_dict
        else:
            state_dict = checkpoint
        logging.info(
            "Loaded {} from checkpoint '{}'".format(state_dict_key, checkpoint_path)
        )
        return state_dict
    else:
        logging.error("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()


def load_checkpoint(model, checkpoint_path, use_ema=False, strict=True):
    state_dict = load_state_dict(checkpoint_path, use_ema)
    model.load_state_dict(state_dict, strict=strict)


# def resume_checkpoint(model, checkpoint_path, optimizer=None, loss_scaler=None, log_info=True):
#     resume_epoch = None
# if os.path.isfile(checkpoint_path):
#     checkpoint = torch.load(checkpoint_path, map_location='cpu')
#     if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
#         if log_info:
#             _logger.info('Restoring model state from checkpoint...')
#         new_state_dict = OrderedDict()
#         for k, v in checkpoint['state_dict'].items():
#             name = k[7:] if k.startswith('module') else k
#             new_state_dict[name] = v
#         model.load_state_dict(new_state_dict)

#         if optimizer is not None and 'optimizer' in checkpoint:
#             if log_info:
#                 _logger.info('Restoring optimizer state from checkpoint...')
#             optimizer.load_state_dict(checkpoint['optimizer'])

#         if loss_scaler is not None and loss_scaler.state_dict_key in checkpoint:
#             if log_info:
#                 _logger.info('Restoring AMP loss scaler state from checkpoint...')
#             loss_scaler.load_state_dict(checkpoint[loss_scaler.state_dict_key])

#         if 'epoch' in checkpoint:
#             resume_epoch = checkpoint['epoch']
#             if 'version' in checkpoint and checkpoint['version'] > 1:
#                 resume_epoch += 1  # start at the next epoch, old checkpoints incremented before save

#         if log_info:
#             _logger.info("Loaded checkpoint '{}' (epoch {})".format(checkpoint_path, checkpoint['epoch']))
#     else:
#         model.load_state_dict(checkpoint)
#         if log_info:
#             _logger.info("Loaded checkpoint '{}'".format(checkpoint_path))
#     return resume_epoch
# else:
#     _logger.error("No checkpoint found at '{}'".format(checkpoint_path))
#     raise FileNotFoundError()


def load_pretrained(
    model,
    cfg=None,
    num_classes=1000,
    in_chans=3,
    filter_fn=None,
    img_size=224,
    num_frames=8,
    num_patches=196,
    attention_type="divided_space_time",
    pretrained_model="",
    strict=True,
):
    if cfg is None:
        cfg = getattr(model, "default_cfg")
    if cfg is None or "url" not in cfg or not cfg["url"]:
        logging.warning("Pretrained model URL is invalid, using random initialization.")
        return

    if len(pretrained_model) == 0:
        if cfg is None:
            logging.info(f"loading from default config {model.default_cfg}.")
        state_dict = model_zoo.load_url(cfg["url"], progress=False, map_location="cpu")
    else:
        try:
            state_dict = load_state_dict(pretrained_model)["model"]
        except:
            state_dict = load_state_dict(pretrained_model)

    if filter_fn is not None:
        state_dict = filter_fn(state_dict)

    if in_chans == 1:
        conv1_name = cfg["first_conv"]
        logging.info(
            "Converting first conv (%s) pretrained weights from 3 to 1 channel"
            % conv1_name
        )
        conv1_weight = state_dict[conv1_name + ".weight"]
        conv1_type = conv1_weight.dtype
        conv1_weight = conv1_weight.float()
        O, I, J, K = conv1_weight.shape
        if I > 3:
            assert conv1_weight.shape[1] % 3 == 0
            # For models with space2depth stems
            conv1_weight = conv1_weight.reshape(O, I // 3, 3, J, K)
            conv1_weight = conv1_weight.sum(dim=2, keepdim=False)
        else:
            conv1_weight = conv1_weight.sum(dim=1, keepdim=True)
        conv1_weight = conv1_weight.to(conv1_type)
        state_dict[conv1_name + ".weight"] = conv1_weight
    elif in_chans != 3:
        conv1_name = cfg["first_conv"]
        conv1_weight = state_dict[conv1_name + ".weight"]
        conv1_type = conv1_weight.dtype
        conv1_weight = conv1_weight.float()
        O, I, J, K = conv1_weight.shape
        if I != 3:
            logging.warning(
                "Deleting first conv (%s) from pretrained weights." % conv1_name
            )
            del state_dict[conv1_name + ".weight"]
            strict = False
        else:
            logging.info(
                "Repeating first conv (%s) weights in channel dim." % conv1_name
            )
            repeat = int(math.ceil(in_chans / 3))
            conv1_weight = conv1_weight.repeat(1, repeat, 1, 1)[:, :in_chans, :, :]
            conv1_weight *= 3 / float(in_chans)
            conv1_weight = conv1_weight.to(conv1_type)
            state_dict[conv1_name + ".weight"] = conv1_weight

    classifier_name = cfg["classifier"]
    if num_classes == 1000 and cfg["num_classes"] == 1001:
        # special case for imagenet trained models with extra background class in pretrained weights
        classifier_weight = state_dict[classifier_name + ".weight"]
        state_dict[classifier_name + ".weight"] = classifier_weight[1:]
        classifier_bias = state_dict[classifier_name + ".bias"]
        state_dict[classifier_name + ".bias"] = classifier_bias[1:]
    elif num_classes != state_dict[classifier_name + ".weight"].size(0):
        # print('Removing the last fully connected layer due to dimensions mismatch ('+str(num_classes)+ ' != '+str(state_dict[classifier_name + '.weight'].size(0))+').', flush=True)
        # completely discard fully connected for all other differences between pretrained and created model
        del state_dict[classifier_name + ".weight"]
        del state_dict[classifier_name + ".bias"]
        strict = False

    ## Resizing the positional embeddings in case they don't match
    logging.info(
        f"Resizing spatial position embedding from {state_dict['pos_embed'].size(1)} to {num_patches + 1}"
    )
    if num_patches + 1 != state_dict["pos_embed"].size(1):
        pos_embed = state_dict["pos_embed"]
        cls_pos_embed = pos_embed[0, 0, :].unsqueeze(0).unsqueeze(1)
        other_pos_embed = pos_embed[0, 1:, :].unsqueeze(0).transpose(1, 2)
        new_pos_embed = F.interpolate(
            other_pos_embed, size=(num_patches), mode="nearest"
        )
        new_pos_embed = new_pos_embed.transpose(1, 2)
        new_pos_embed = torch.cat((cls_pos_embed, new_pos_embed), 1)
        state_dict["pos_embed"] = new_pos_embed

    ## Resizing time embeddings in case they don't match
    if "time_embed" in state_dict and num_frames != state_dict["time_embed"].size(1):
        logging.info(
            f"Resizing temporal position embedding from {state_dict['time_embed'].size(1)} to {num_frames}"
        )
        time_embed = state_dict["time_embed"].transpose(1, 2)
        new_time_embed = F.interpolate(time_embed, size=(num_frames), mode="nearest")
        state_dict["time_embed"] = new_time_embed.transpose(1, 2)

    ## Initializing temporal attention
    if attention_type == "divided_space_time":
        new_state_dict = state_dict.copy()
        for key in state_dict:
            if "blocks" in key and "attn" in key:
                new_key = key.replace("attn", "temporal_attn")
                if not new_key in state_dict:
                    new_state_dict[new_key] = state_dict[key]
                else:
                    new_state_dict[new_key] = state_dict[new_key]
            if "blocks" in key and "norm1" in key:
                new_key = key.replace("norm1", "temporal_norm1")
                if not new_key in state_dict:
                    new_state_dict[new_key] = state_dict[key]
                else:
                    new_state_dict[new_key] = state_dict[new_key]
        state_dict = new_state_dict

    ## Loading the weights
    model.load_state_dict(state_dict, strict=False)


def load_pretrained_imagenet(
    model,
    pretrained_model,
    cfg=None,
    ignore_classifier=True,
    num_frames=8,
    num_patches=196,
    **kwargs,
):
    import timm

    logging.info(f"Loading vit_base_patch16_224 checkpoints.")
    loaded_state_dict = timm.models.vision_transformer.vit_base_patch16_224(
        pretrained=True
    ).state_dict()

    del loaded_state_dict["head.weight"]
    del loaded_state_dict["head.bias"]

    ## Initializing temporal attention
    new_state_dict = loaded_state_dict.copy()
    for key in loaded_state_dict:
        if "blocks" in key and "attn" in key:
            new_key = key.replace("attn", "temporal_attn")
            if not new_key in loaded_state_dict:
                new_state_dict[new_key] = loaded_state_dict[key]
            else:
                new_state_dict[new_key] = loaded_state_dict[new_key]
        if "blocks" in key and "norm1" in key:
            new_key = key.replace("norm1", "temporal_norm1")
            if not new_key in loaded_state_dict:
                new_state_dict[new_key] = loaded_state_dict[key]
            else:
                new_state_dict[new_key] = loaded_state_dict[new_key]

    loaded_state_dict = new_state_dict

    loaded_keys = loaded_state_dict.keys()
    model_keys = model.state_dict().keys()

    load_not_in_model = [k for k in loaded_keys if k not in model_keys]
    model_not_in_load = [k for k in model_keys if k not in loaded_keys]

    toload = dict()
    mismatched_shape_keys = []
    for k in model_keys:
        if k in loaded_keys:
            if model.state_dict()[k].shape != loaded_state_dict[k].shape:
                mismatched_shape_keys.append(k)
            else:
                toload[k] = loaded_state_dict[k]

    logging.info("Keys in loaded but not in model:")
    logging.info(f"In total {len(load_not_in_model)}, {sorted(load_not_in_model)}")
    logging.info("Keys in model but not in loaded:")
    logging.info(f"In total {len(model_not_in_load)}, {sorted(model_not_in_load)}")
    logging.info("Keys in model and loaded, but shape mismatched:")
    logging.info(
        f"In total {len(mismatched_shape_keys)}, {sorted(mismatched_shape_keys)}"
    )

    model.load_state_dict(toload, strict=False)


def load_pretrained_kinetics(
    model,
    pretrained_model,
    cfg=None,
    ignore_classifier=True,
    num_frames=8,
    num_patches=196,
    **kwargs,
):
    if cfg is None:
        cfg = getattr(model, "default_cfg")
    if cfg is None or "url" not in cfg or not cfg["url"]:
        logging.warning("Pretrained model URL is invalid, using random initialization.")
        return

    assert (
        len(pretrained_model) > 0
    ), "Path to pre-trained Kinetics weights not provided."

    state_dict = load_state_dict(pretrained_model)

    classifier_name = cfg["classifier"]
    if ignore_classifier:

        classifier_weight_key = classifier_name + ".weight"
        classifier_bias_key = classifier_name + ".bias"

        state_dict[classifier_weight_key] = model.state_dict()[classifier_weight_key]
        state_dict[classifier_bias_key] = model.state_dict()[classifier_bias_key]

    else:
        raise NotImplementedError(
            "[dxli] Not supporting loading Kinetics-pretrained ckpt with classifier."
        )

    ## Resizing the positional embeddings in case they don't match
    if num_patches + 1 != state_dict["pos_embed"].size(1):
        new_pos_embed = resize_spatial_embedding(state_dict, "pos_embed", num_patches)
        state_dict["pos_embed"] = new_pos_embed

    ## Resizing time embeddings in case they don't match
    if "time_embed" in state_dict and num_frames != state_dict["time_embed"].size(1):
        state_dict["time_embed"] = resize_temporal_embedding(
            state_dict, "time_embed", num_frames
        )

    ## Loading the weights
    try:
        model.load_state_dict(state_dict, strict=True)
        logging.info("Succeeded in loading Kinetics pre-trained weights.")
    except:
        logging.error("Error in loading Kinetics pre-trained weights.")


def resize_spatial_embedding(state_dict, key, num_patches):
    logging.info(
        f"Resizing spatial position embedding from {state_dict[key].size(1)} to {num_patches + 1}"
    )

    pos_embed = state_dict[key]

    cls_pos_embed = pos_embed[0, 0, :].unsqueeze(0).unsqueeze(1)
    other_pos_embed = pos_embed[0, 1:, :].unsqueeze(0).transpose(1, 2)

    new_pos_embed = F.interpolate(other_pos_embed, size=(num_patches), mode="nearest")
    new_pos_embed = new_pos_embed.transpose(1, 2)
    new_pos_embed = torch.cat((cls_pos_embed, new_pos_embed), 1)

    return new_pos_embed


def resize_temporal_embedding(state_dict, key, num_frames):
    logging.info(
        f"Resizing temporal position embedding from {state_dict[key].size(1)} to {num_frames}"
    )

    time_embed = state_dict[key].transpose(1, 2)
    new_time_embed = F.interpolate(time_embed, size=(num_frames), mode="nearest")

    return new_time_embed.transpose(1, 2)


def detach_variable(inputs):
    if isinstance(inputs, tuple):
        out = []
        for inp in inputs:
            x = inp.detach()
            x.requires_grad = inp.requires_grad
            out.append(x)
        return tuple(out)
    else:
        raise RuntimeError(
            "Only tuple of tensors is supported. Got Unsupported input type: ",
            type(inputs).__name__,
        )


def check_backward_validity(inputs):
    if not any(inp.requires_grad for inp in inputs):
        warnings.warn(
            "None of the inputs have requires_grad=True. Gradients will be None"
        )
