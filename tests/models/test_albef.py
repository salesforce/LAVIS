"""
Integration tests for ALBEF models.
"""

import pytest
import torch
from lavis.models import load_model, load_model_and_preprocess
from PIL import Image

# setup device to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load sample image
raw_image = Image.open("docs/data/merlion.png").convert("RGB")

precision = 1e-3


class TestAlbef:
    def test_vqa(self):
        model, vis_processors, txt_processors = load_model_and_preprocess(
            name="albef_vqa", model_type="vqav2", is_eval=True, device=device
        )

        # ask a random question.
        question = "Which city is this photo taken?"

        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
        question = txt_processors["eval"](question)

        samples = {"image": image, "text_input": question}

        answer_list = ["Singapore", "London", "Palo Alto", "Tokyo"]
        answers = model.predict_answers(samples, answer_list=answer_list)

        assert answers == ["Singapore"]

    def test_vqa_forward(self):
        model = load_model("albef_vqa")
        samples = {
            "image": torch.rand(2, 3, 384, 384),
            "text_input": ["What is this?", "What is that?"],
            "answer": ["cat", "cat", "dog"],
            "weight": torch.tensor([1.0, 1.0, 1.0]),
            "n_answers": torch.tensor([2, 1]),
            "epoch": 0,
            "iters": 0,
            "num_iters_per_epoch": 1000,
        }
        output = model(samples)
        # odict_keys(['image_embeds', 'image_embeds_m', 'encoder_output', 'encoder_output_m', 'decoder_output', 'decoder_labels'])

        assert output.intermediate_output.image_embeds.shape == torch.Size(
            [2, 577, 768]
        )
        assert output.intermediate_output.image_embeds_m.shape == torch.Size(
            [2, 577, 768]
        )

        assert (
            output.intermediate_output.encoder_output.last_hidden_state.shape
            == torch.Size([2, 6, 768])
        )
        assert (
            output.intermediate_output.encoder_output_m.last_hidden_state.shape
            == torch.Size([2, 6, 768])
        )

        assert output.intermediate_output.decoder_output.logits.shape == torch.Size(
            [3, 3, 30522]
        )
        assert output.intermediate_output.decoder_labels.shape == torch.Size([3, 3])

    def test_retrieval(self):
        model = load_model("albef_retrieval", "coco", is_eval=True, device=device)

        images = torch.randn(4, 3, 384, 384).to(device)
        text_input = [
            "caption of image 1",
            "another caption of image 1",
            "caption of image 2",
            "caption of image 3",
        ]
        image_id = torch.LongTensor([1, 1, 2, 3]).to(device)
        samples = {
            "image": images,
            "text_input": text_input,
            "image_id": image_id,
            "epoch": 0,
            "iters": 0,
            "num_iters_per_epoch": 100,
        }
        output = model(samples)

        assert output.intermediate_output.image_embeds.shape == torch.Size(
            [4, 577, 768]
        )
        assert output.intermediate_output.text_embeds.shape == torch.Size([4, 30, 768])
        assert output.intermediate_output.image_embeds_m.shape == torch.Size(
            [4, 577, 768]
        )
        assert output.intermediate_output.text_embeds_m.shape == torch.Size(
            [4, 30, 768]
        )
        assert (
            output.intermediate_output.encoder_output.last_hidden_state.shape
            == torch.Size([4, 30, 768])
        )
        assert output.intermediate_output.itm_logits.shape == torch.Size([12, 2])
        assert all(
            output.intermediate_output.itm_labels
            == torch.LongTensor([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]).to(device)
        )

    def test_pretrain(self):
        model = load_model("albef_pretrain", is_eval=True, device=device)

        images = torch.randn(4, 3, 224, 224).to(device)

        text_input = [
            "caption of image 1",
            "another caption of image 1",
            "caption of image 2",
            "caption of image 3",
        ]
        samples = {
            "image": images,
            "text_input": text_input,
            "epoch": 0,
            "iters": 0,
            "num_iters_per_epoch": 100,
        }
        output = model(samples)

        assert output.intermediate_output.image_embeds.shape == torch.Size(
            [4, 197, 768]
        )
        assert output.intermediate_output.text_embeds.shape == torch.Size([4, 30, 768])
        assert output.intermediate_output.itm_logits.shape == torch.Size([12, 2])
        assert all(
            output.intermediate_output.itm_labels
            == torch.LongTensor([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]).to(device)
        )

    # def test_feature_extractor(self):
    #     from PIL import Image
    #     from lavis.models import load_model_and_preprocess

    #     raw_image = Image.open("docs/data/merlion.png").convert("RGB")
    #     caption = "a large fountain spewing water into the air"

    #     model, vis_processors, txt_processors = load_model_and_preprocess(
    #         "albef_feature_extractor", is_eval=True
    #     )

    #     image = vis_processors["eval"](raw_image).unsqueeze(0)
    #     text_input = txt_processors["eval"](caption)

    #     sample = {"image": image, "text_input": [text_input]}

    #     features_multimodal = model.extract_features(sample)
    #     features_text = model.extract_features(sample, mode="text")
    #     features_image = model.extract_features(sample, mode="image")

    #     assert features_multimodal.image_embeds.shape == torch.Size([1, 197, 768])
    #     assert features_multimodal.multimodal_embeds.shape == torch.Size([1, 12, 768])

    #     assert features_text.text_embeds.shape == torch.Size([1, 12, 768])
    #     assert features_text.text_features.shape == torch.Size([1, 12, 256])

    #     assert features_image.image_embeds.shape == torch.Size([1, 197, 768])
    #     assert features_image.image_features.shape == torch.Size([1, 197, 256])

    # assert torch.sum(features_multimodal.image_embeds).item() == pytest.approx(
    #     -3074.2212, precision
    # )
    # assert torch.sum(features_multimodal.multimodal_embeds).item() == pytest.approx(
    #     -8.7235, precision
    # )

    # assert torch.sum(features_text.text_embeds).item() == pytest.approx(
    #     -0.6083, precision
    # )
    # assert torch.sum(features_text.text_features).item() == pytest.approx(
    #     -6.6052, precision
    # )

    # assert torch.sum(features_image.image_embeds).item() == pytest.approx(
    #     -3074.2212, precision
    # )
    # assert torch.sum(features_image.image_features).item() == pytest.approx(
    #     -116.0622, precision
    # )

    # def test_itm(self):
    #     from lavis.processors.blip_processors import BlipCaptionProcessor

    #     vis_processor = load_processor("blip_image_eval").build(image_size=384)

    #     text_processor = BlipCaptionProcessor(prompt="A picture of ")
    #     cls_prompt = [
    #         text_processor(cls_nm)
    #         for cls_nm in ["merlion", "elephant", "giraffe", "fountain", "marina bay"]
    #     ]

    #     image = vis_processor(raw_image).unsqueeze(0).to(device)

    #     model = load_blip_itm_model(device)

    #     output = model(image, cls_prompt, match_head="itm")
    #     sims = output[:, 1]

    #     sims = torch.nn.Softmax(dim=0)(sims)
    #     inv_sims = [sim * 100 for sim in sims.tolist()[::-1]]

    #     print(inv_sims)
