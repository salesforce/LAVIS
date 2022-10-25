"""
#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#

Integration tests for PNP-VQA model.
"""

import pytest
import torch
from lavis.models import load_model, load_model_and_preprocess
from PIL import Image

# setup device to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load sample image
raw_image = Image.open("docs/_static/merlion.png").convert("RGB")

precision = 1e-1


class TestPNPVQA:
    def test_vqa(self):
        # loads PNP-VQA base model, with BLIP_itm_large, BLIP_caption_large, Unifiedqav2_base
        # this also loads the associated image processors and text processors
        model, vis_processors, txt_processors = load_model_and_preprocess(
            name="pnp_vqa", model_type="base", is_eval=True, device=device
        )

        # ask a random question.
        question = "Which city is this photo taken?"

        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
        question = txt_processors["eval"](question)

        samples = {"image": image, "text_input": [question]}

        answer, caption, gradcam = model.predict_answers(
            samples=samples,
            inference_method="generate",
            num_captions=5,
            num_patches=20,
        )
        assert isinstance(answer, list)
        assert isinstance(caption, list)
        assert isinstance(gradcam, torch.Tensor)

        assert len(answer) == 1
        assert len(caption) == 1
        assert len(caption[0]) == 5
        assert gradcam.size() == torch.Size([1,576])

    def test_itm(self):
        # loads PNP-VQA base model, with BLIP_itm_large, BLIP_caption_large, Unifiedqav2_base
        # this also loads the associated image processors and text processors
        model, vis_processors, txt_processors = load_model_and_preprocess(
            name="pnp_vqa", model_type="base", is_eval=True, device=device
        )

        # ask a random question.
        question = "Which city is this photo taken?"

        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
        question = txt_processors["eval"](question)

        samples = {"image": image, "text_input": [question]}

        samples = model.forward_itm(samples=samples)

        assert samples['gradcams'].size() == torch.Size([1,576])

    def test_caption(self):
        # loads PNP-VQA base model, with BLIP_itm_large, BLIP_caption_large, Unifiedqav2_base
        # this also loads the associated image processors and text processors
        model, vis_processors, txt_processors = load_model_and_preprocess(
            name="pnp_vqa", model_type="base", is_eval=True, device=device
        )

        # ask a random question.
        question = "Which city is this photo taken?"

        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
        question = txt_processors["eval"](question)

        samples = {"image": image, "text_input": [question]}

        samples['gradcams'] = torch.rand(1,576)

        samples = model.forward_cap(samples=samples, num_captions=5, num_patches=20)

        assert len(samples['captions']) == 1
        assert len(samples['captions'][0]) == 5

    def test_qa(self):
        # loads PNP-VQA base model, with BLIP_itm_large, BLIP_caption_large, Unifiedqav2_base
        # this also loads the associated image processors and text processors
        model, vis_processors, txt_processors = load_model_and_preprocess(
            name="pnp_vqa", model_type="base", is_eval=True, device=device
        )

        # ask a random question.
        question = "Which city is this photo taken?"

        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
        question = txt_processors["eval"](question)

        samples = {"image": image, "text_input": [question]}

        samples['captions'] = [['the city is singapore', 'the picture is taken in singapore']]

        answer = model.forward_qa(samples=samples, num_captions=2)

        assert isinstance(answer, list)
        assert len(answer) == 1
        assert answer[0]== 'singapore'