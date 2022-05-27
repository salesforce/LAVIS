import logging
import os

import torch
from common.utils import is_url
from models.vit import interpolate_pos_embed
from timm.models.hub import download_cached_file
from transformers import BertTokenizer


def init_tokenizer():
    return BertTokenizer.from_pretrained("bert-base-uncased")


def load_from_pretrained(model, url_or_filename):
    if is_url(url_or_filename):
        cached_file = download_cached_file(
            url_or_filename, check_hash=False, progress=True
        )
        checkpoint = torch.load(cached_file, map_location="cpu")
    elif os.path.isfile(url_or_filename):
        checkpoint = torch.load(url_or_filename, map_location="cpu")
    else:
        raise RuntimeError("checkpoint url or path is invalid")

    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    state_dict["visual_encoder.pos_embed"] = interpolate_pos_embed(
        state_dict["visual_encoder.pos_embed"], model.visual_encoder
    )
    if (
        "visual_encoder_m.pos_embed" in model.state_dict().keys()
        and "visual_encoder_m.pos_embed" in state_dict
    ):
        state_dict["visual_encoder_m.pos_embed"] = interpolate_pos_embed(
            state_dict["visual_encoder_m.pos_embed"], model.visual_encoder_m
        )

    for key in list(state_dict.keys()):
        if "bert" in key:
            new_key = key.replace("bert.", "")
            state_dict[new_key] = state_dict[key]
            del state_dict[key]

    for key in model.state_dict().keys():
        if key in state_dict.keys():
            if state_dict[key].shape != model.state_dict()[key].shape:
                del state_dict[key]

    msg = model.load_state_dict(state_dict, strict=False)

    logging.info("Missing keys {}".format(msg.missing_keys))
    logging.info("load checkpoint from %s" % url_or_filename)
    return model, msg
