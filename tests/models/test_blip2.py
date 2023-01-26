"""
#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#

Integration tests for BLIP2 models.
"""

import pytest
import torch
from lavis.models import load_model, load_model_and_preprocess
from PIL import Image

# setup device to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load sample image
raw_image = Image.open("docs/_static/merlion.png").convert("RGB")


class TestBlip2:
    def test_blip2_opt2p7b(self):
        # loads BLIP2-OPT-2.7b caption model, without finetuning on coco.
        model, vis_processors, _ = load_model_and_preprocess(
            name="blip2_opt", model_type="pretrain_opt2.7b", is_eval=True, device=device
        )

        # preprocess the image
        # vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)
        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

        # generate caption
        caption = model.generate({"image": image})

        assert caption == ["the merlion fountain in singapore"]

        # generate multiple captions
        captions = model.generate({"image": image}, num_captions=3)

        assert len(captions) == 3

    def test_blip2_opt2p7b_coco(self):
        # loads BLIP2-OPT-2.7b caption model,
        model, vis_processors, _ = load_model_and_preprocess(
            name="blip2_opt",
            model_type="caption_coco_opt2.7b",
            is_eval=True,
            device=device,
        )

        # preprocess the image
        # vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)
        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

        # generate caption
        caption = model.generate({"image": image})

        assert caption == ["a statue of a mermaid spraying water into the air"]

        # generate multiple captions
        captions = model.generate({"image": image}, num_captions=3)

        assert len(captions) == 3

    def test_blip2_opt6p7b(self):
        # loads BLIP2-OPT-2.7b caption model,
        model, vis_processors, _ = load_model_and_preprocess(
            name="blip2_opt", model_type="pretrain_opt6.7b", is_eval=True, device=device
        )

        # preprocess the image
        # vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)
        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

        # generate caption
        caption = model.generate({"image": image})

        assert caption == ["a statue of a merlion in front of a water fountain"]

        # generate multiple captions
        captions = model.generate({"image": image}, num_captions=3)

        assert len(captions) == 3

    def test_blip2_opt6p7b_coco(self):
        # loads BLIP2-OPT-2.7b caption model,
        model, vis_processors, _ = load_model_and_preprocess(
            name="blip2_opt",
            model_type="caption_coco_opt6.7b",
            is_eval=True,
            device=device,
        )

        # preprocess the image
        # vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)
        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

        # generate caption
        caption = model.generate({"image": image})

        assert caption == ["a large fountain spraying water into the air"]

        # generate multiple captions
        captions = model.generate({"image": image}, num_captions=3)

        assert len(captions) == 3

    def test_blip2_flant5xl(self):
        # loads BLIP2-FLAN-T5XL caption model,
        model, vis_processors, _ = load_model_and_preprocess(
            name="blip2_t5", model_type="pretrain_flant5xl", is_eval=True, device=device
        )

        # preprocess the image
        # vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)
        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

        # generate caption
        caption = model.generate({"image": image})

        assert caption == ["marina bay sands, singapore"]

        # generate multiple captions
        captions = model.generate({"image": image}, num_captions=3)

        assert len(captions) == 3

    def test_blip2_flant5xxl(self):
        # loads BLIP2-FLAN-T5XXL caption model,
        model, vis_processors, _ = load_model_and_preprocess(
            name="blip2_t5",
            model_type="pretrain_flant5xxl",
            is_eval=True,
            device=device,
        )

        # preprocess the image
        # vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)
        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

        # generate caption
        caption = model.generate({"image": image})

        assert caption == ["the merlion statue in singapore"]

        # generate multiple captions
        captions = model.generate({"image": image}, num_captions=3)

        assert len(captions) == 3
