"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
import os

import torch
import torch.nn.functional as F
from lavis.common.dist_utils import download_cached_file
from lavis.common.utils import is_url
from lavis.models.base_model import BaseModel
from transformers import BertTokenizer


class AlproBase(BaseModel):
    @classmethod
    def init_tokenizer(cls):
        return BertTokenizer.from_pretrained("bert-base-uncased")

    def load_from_pretrained(self, url_or_filename, num_frames, num_patches):
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

        for key in list(state_dict.keys()):
            if "bert" in key:
                new_key = key.replace("bert.", "")
                state_dict[new_key] = state_dict[key]
                del state_dict[key]

        spatial_embed_key = "visual_encoder.model.pos_embed"
        temporal_embed_key = "visual_encoder.model.time_embed"

        ## Resizing spatial embeddings in case they don't match
        if num_patches + 1 != state_dict[spatial_embed_key].size(1):
            state_dict[spatial_embed_key] = resize_spatial_embedding(
                state_dict, spatial_embed_key, num_patches
            )
        else:
            logging.info(
                "The length of spatial position embedding matches. No need to resize."
            )

        ## Resizing time embeddings in case they don't match
        if temporal_embed_key in state_dict and num_frames != state_dict[
            temporal_embed_key
        ].size(1):
            state_dict[temporal_embed_key] = resize_temporal_embedding(
                state_dict, temporal_embed_key, num_frames
            )
        else:
            logging.info(
                "No temporal encoding found. Or the length of temporal position embedding matches. No need to resize."
            )

        msg = self.load_state_dict(state_dict, strict=False)
        logging.info("Missing keys {}".format(msg.missing_keys))
        logging.info("load checkpoint from %s" % url_or_filename)

        return msg


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
