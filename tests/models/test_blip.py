"""
Integration tests for BLIP models.
"""

import pytest
import torch
from lavis.models import load_model, load_model_and_preprocess
from lavis.models.blip_models.blip import BlipITM
from lavis.processors import load_processor
from PIL import Image

# setup device to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load sample image
raw_image = Image.open("docs/data/merlion.png").convert("RGB")

precision = 1e-4


def load_blip_itm_model(device):
    pretrained_path = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_retrieval_coco.pth"
    model = BlipITM(pretrained=pretrained_path, vit="base")
    model.eval()
    model = model.to(device)
    return model


class TestBlip:
    def test_caption(self):
        # loads BLIP caption base model, with finetuned checkpoints on MSCOCO captioning dataset.
        # this also loads the associated image processors
        model, vis_processors, _ = load_model_and_preprocess(
            name="blip_caption", model_type="base_coco", is_eval=True, device=device
        )

        # preprocess the image
        # vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)
        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

        # generate caption
        caption = model.generate({"image": image})

        assert caption == ["a large fountain spewing water into the air"]

        # generate multiple captions
        captions = model.generate({"image": image}, num_captions=3)

        assert len(captions) == 3

    def test_caption_large(self):
        # loads BLIP caption base model, with finetuned checkpoints on MSCOCO captioning dataset.
        # this also loads the associated image processors
        model, vis_processors, _ = load_model_and_preprocess(
            name="blip_caption", model_type="large_coco", is_eval=True, device=device
        )

        # preprocess the image
        # vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)
        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

        # generate caption
        caption = model.generate({"image": image})

        assert caption == ["a large statue of a person spraying water from a fountain"]

        # generate multiple captions
        captions = model.generate({"image": image}, num_captions=3)

        assert len(captions) == 3

    def test_caption_forward(self):
        model, vis_processors, _ = load_model_and_preprocess(
            name="blip_caption", model_type="base_coco", is_eval=True, device=device
        )

        # preprocess the image
        # vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)
        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
        text_input = ["a large statue of a person spraying water from a fountain"]

        samples = {"image": image, "text_input": text_input}
        output = model(samples)

        assert output.intermediate_output.image_embeds.shape == torch.Size(
            [1, 577, 768]
        )
        assert output.intermediate_output.decoder_labels.shape == torch.Size([1, 13])

        assert pytest.approx(2.7152, precision) == output.loss.item()
        assert (
            pytest.approx(-8863.6962, precision)
            == torch.sum(output.intermediate_output.image_embeds).item()
        )

        assert all(
            output.intermediate_output.decoder_labels[0]
            == torch.LongTensor(
                [
                    -100,
                    -100,
                    -100,
                    -100,
                    1997,
                    1037,
                    2711,
                    29035,
                    2300,
                    2013,
                    1037,
                    9545,
                    102,
                ]
            ).to(device)
        )

    def test_vqa(self):
        model, vis_processors, txt_processors = load_model_and_preprocess(
            name="blip_vqa", model_type="vqav2", is_eval=True, device=device
        )

        # ask a random question.
        question = "Which city is this photo taken?"

        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
        question = txt_processors["eval"](question)

        samples = {"image": image, "text_input": question}

        answer = model.predict_answers(
            samples=samples,
            inference_method="generate",
        )
        assert answer == ["singapore"]

        answer_list = ["Singapore", "London", "Palo Alto", "Tokyo"]
        answers = model.predict_answers(samples, answer_list=answer_list)

        assert answers == ["Singapore"]

    def test_retrieval(self):
        model = load_model("blip_retrieval", "coco")

        images = torch.randn(4, 3, 384, 384)
        text_input = [
            "caption of image 1",
            "another caption of image 1",
            "caption of image 2",
            "caption of image 3",
        ]
        image_id = torch.tensor([1, 1, 2, 3])
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
        assert output.intermediate_output.text_embeds.shape == torch.Size([4, 35, 768])
        assert output.intermediate_output.image_embeds_m.shape == torch.Size(
            [4, 577, 768]
        )
        assert output.intermediate_output.text_embeds_m.shape == torch.Size(
            [4, 35, 768]
        )
        assert (
            output.intermediate_output.encoder_output.last_hidden_state.shape
            == torch.Size([4, 35, 768])
        )
        assert output.intermediate_output.itm_logits.shape == torch.Size([12, 2])
        assert all(
            output.intermediate_output.itm_labels
            == torch.LongTensor([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        )

    def test_pretrain(self):
        model = load_model("blip_pretrain", "base")

        images = torch.randn(4, 3, 224, 224)

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
            == torch.LongTensor([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        )
        assert output.intermediate_output.decoder_labels.shape == torch.Size([4, 30])
        assert output.intermediate_output.decoder_output.logits.shape == torch.Size(
            [4, 30, 30524]
        )

    def test_feature_extractor(self):
        from PIL import Image
        from lavis.models import load_model_and_preprocess

        raw_image = Image.open("docs/data/merlion.png").convert("RGB")
        caption = "a large fountain spewing water into the air"

        model, vis_processors, txt_processors = load_model_and_preprocess(
            "blip_feature_extractor", is_eval=True
        )

        image = vis_processors["eval"](raw_image).unsqueeze(0)
        text_input = txt_processors["eval"](caption)

        sample = {"image": image, "text_input": [text_input]}

        features_multimodal = model.extract_features(sample)
        features_text = model.extract_features(sample, mode="text")
        features_image = model.extract_features(sample, mode="image")

        assert features_multimodal.image_embeds.shape == torch.Size([1, 197, 768])
        assert features_multimodal.multimodal_embeds.shape == torch.Size([1, 12, 768])

        assert features_text.text_embeds.shape == torch.Size([1, 12, 768])
        assert features_text.text_features.shape == torch.Size([1, 12, 256])

        assert features_image.image_embeds.shape == torch.Size([1, 197, 768])
        assert features_image.image_features.shape == torch.Size([1, 197, 256])

        assert torch.sum(features_multimodal.image_embeds).item() == pytest.approx(
            -3074.2212, precision
        )
        assert torch.sum(features_multimodal.multimodal_embeds).item() == pytest.approx(
            -8.7235, precision
        )

        assert torch.sum(features_text.text_embeds).item() == pytest.approx(
            -0.6083, precision
        )
        assert torch.sum(features_text.text_features).item() == pytest.approx(
            -6.6052, precision
        )

        assert torch.sum(features_image.image_embeds).item() == pytest.approx(
            -3074.2212, precision
        )
        assert torch.sum(features_image.image_features).item() == pytest.approx(
            -116.0622, precision
        )

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
