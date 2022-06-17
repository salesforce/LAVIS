import logging
import os
import torch
import torch.nn.functional as F

from transformers import BertTokenizer

from lavis.common.utils import is_url
from timm.models.hub import download_cached_file


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

    # for key in list(state_dict.keys()):
    #     if "visual_encoder.model" in key:
    #         new_key = key.replace("visual_encoder", "")
    #         state_dict[new_key] = state_dict[key]
    #         del state_dict[key]

    ## Resizing time embeddings in case they don't match
    # num_frames = 16
    # if "time_embed" in state_dict and num_frames != state_dict["time_embed"].size(1):
    #     logging.info(
    #         f"Resizing temporal position embedding from {state_dict['time_embed'].size(1)} to {num_frames}"
    #     )
    #     time_embed = state_dict["time_embed"].transpose(1, 2)
    #     new_time_embed = F.interpolate(time_embed, size=(num_frames), mode="nearest")
    #     state_dict["time_embed"] = new_time_embed.transpose(1, 2)

    msg = model.load_state_dict(state_dict, strict=False)
    logging.info("Missing keys {}".format(msg.missing_keys))
    logging.info("load checkpoint from %s" % url_or_filename)

    return model, msg
