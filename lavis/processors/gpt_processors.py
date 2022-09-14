"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import re

from lavis.common.registry import registry
from lavis.processors.base_processor import BaseProcessor
from lavis.processors.randaugment import RandomAugment
from omegaconf import OmegaConf
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import os
from itertools import chain
import numpy as np
import torch
from transformers import GPT2Tokenizer

SPECIAL_TOKENS_DICT = {
    "bos_token": "<bos>",
    "eos_token": "<eos>",
    "additional_special_tokens": ["<speaker1>", "<speaker2>", "<video>", "<cap>"],
    "pad_token": "<pad>",
}
SPECIAL_TOKENS = [
    "<bos>",
    "<eos>",
    "<speaker1>",
    "<speaker2>",
    "<cap>",
    "<video>",
    "<pad>",
]


class GPTVideoFeatureBaseProcessor(BaseProcessor):
    def __init__(self, visual_ft=["i3d_rgb"], audio_ft=["vggish"]):
        self.visual_ft = visual_ft
        self.audio_ft = audio_ft


@registry.register_processor("gpt_dialogue")
class GPTDialogueProcessor(BaseProcessor):
    def __init__(self, max_turns=3, use_caption=True):
        self.max_turns = max_turns
        self.use_caption = use_caption
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT)

    def sample_sequence(self, caption, history, answer):
        bos, eos, speaker1, speaker2, cap = self.tokenizer.convert_tokens_to_ids(
            SPECIAL_TOKENS[:-2]
        )
        instance = {}
        sequence = [caption] + history + [answer]
        sequence = [s + [eos] for s in sequence]

        instance["input_ids"] = list(chain(*sequence))
        instance["token_type_ids"] = [cap] * len(sequence[0]) + [
            speaker2 if i % 2 else speaker1
            for i, s in enumerate(sequence[1:])
            for _ in s
        ]
        instance["labels"] = ([-1] * sum(len(s) for s in sequence[:-1])) + sequence[-1]

        assert len(instance["input_ids"]) == len(instance["token_type_ids"])
        assert len(instance["token_type_ids"]) == len(instance["labels"])

        for k, v in instance.items():
            instance[k] = torch.Tensor(v).long()

        return instance

    def padding(self, seq, pad_token=-1):
        if pad_token == -1:
            pad_token = self.tokenizer.pad_token_id
        padded_seq = torch.nn.utils.rnn.pad_sequence(
            seq, batch_first=True, padding_value=pad_token
        )
        return padded_seq

    def get_attention_mask(self, seq, pad_token=-1):
        if pad_token == -1:
            pad_token = self.tokenizer.pad_token_id
        return seq != pad_token

    def __call__(self, ann):
        if self.use_caption:
            caption = " ".join([ann["caption"], ann["summary"]])
            caption = self.tokenizer.encode(caption)
        else:
            caption = []

        dial_history = []
        for turn in ann["dialog"][-self.max_turns :]:
            dial_history.append(turn["question"])
            dial_history.append(turn["answer"])
        dial_history.append(ann["question"])
        dial_history = [self.tokenizer.encode(t) for t in dial_history]

        answer = self.tokenizer.encode(ann["answer"])

        item = self.sample_sequence(caption, dial_history, answer)

        return item

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        use_caption = cfg.get("use_caption", True)
        max_turns = cfg.get("max_turns", 3)

        return cls(max_turns=max_turns, use_caption=use_caption)


@registry.register_processor("gpt_video_ft")
class GPTVideoFeatureProcessor(GPTVideoFeatureBaseProcessor):
    def __init__(self, visual_ft, audio_ft):
        super().__init__(visual_ft, audio_ft)
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT)

    def padding(self, seq):
        padded_seq = torch.nn.utils.rnn.pad_sequence(
            seq, batch_first=True, padding_value=1.0
        )
        return padded_seq

    def get_attention_mask(self, seq):
        return torch.sum(seq != 1, dim=2) != 0

    def __call__(self, ft_root, vname):
        all_ft = []

        for ft_name in self.visual_ft:
            ft_path = os.path.join(ft_root, ft_name, vname)
            all_ft.append(np.load(ft_path + ".npy"))

        for ft_name in self.audio_ft:
            ft_path = os.path.join(ft_root, ft_name, vname)
            all_ft.append(np.load(ft_path + ".npy"))

        min_len = min([len(ft) for ft in all_ft])

        # TODO: use other sampling method (e.g. uniform sampling)
        sampled_ft = [ft[:min_len] for ft in all_ft]
        sampled_ft = np.concatenate(sampled_ft, axis=1)
        item = {}
        item["video_fts"] = torch.Tensor(sampled_ft)

        video_type_token = self.tokenizer.convert_tokens_to_ids("<video>")
        item["token_type_ids"] = torch.Tensor(
            [video_type_token] * len(sampled_ft)
        ).long()

        return item

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        visual_ft = cfg.get("visual_ft", ["i3d_rgb"])
        audio_ft = cfg.get("audio_ft", ["vggish"])

        return cls(visual_ft=visual_ft, audio_ft=audio_ft)
