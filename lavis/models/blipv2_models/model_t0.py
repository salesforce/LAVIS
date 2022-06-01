import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
import transformers
from lavis.common.registry import registry
from lavis.models.base_model import BaseModel
from lavis.models.t5 import T5ForConditionalGeneration
from timm.models.vision_transformer import (
    vit_base_patch16_224_in21k,
    vit_large_patch16_224_in21k,
)
from torch import nn
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "true"


@registry.register_model("blipv2_t0")
class BLIPv2_T0(BaseModel):
    def __init__(self, vit="base", lm="bigscience/T0_3B", prompts=["", ""]):
        super().__init__()
        if vit == "base":
            self.visual_encoder = vit_base_patch16_224_in21k(
                pretrained=True, num_classes=0
            )
        elif vit == "large":
            self.visual_encoder = vit_large_patch16_224_in21k(
                pretrained=True, num_classes=0
            )

        self.tokenizer = AutoTokenizer.from_pretrained(lm)
        self.text_model = T5ForConditionalGeneration.from_pretrained(lm)
        self.text_model = self.text_model.bfloat16()
        for param in self.text_model.parameters():
            param.requires_grad = False

        self.vision_to_text = nn.Linear(
            self.visual_encoder.embed_dim, self.text_model.config.hidden_size
        )

        self.prompts = prompts

    @classmethod
    def default_config_path(cls, model_type="base"):
        paths = {
            "base": "lavis/configs/models/blipv2_t0.yaml",
            # "large": "lavis/configs/models/blip_vqa_large.yaml"
        }

        assert model_type in paths, "Unknown model type {}".format(model_type)
        return paths[model_type]

    def encode_image(self, image, pool_size=2):
        image_embeds = self.visual_encoder.forward_features(image)
        cls_embed, patch_embeds = image_embeds[:, :1, :], image_embeds[:, 1:, :]

        bsz, num_patches, dim = patch_embeds.size()
        patch_embeds = patch_embeds.view(
            bsz, int(num_patches**0.5), int(num_patches**0.5), dim
        )
        patch_embeds = patch_embeds.permute(0, 3, 1, 2)
        if pool_size > 1:
            patch_embeds = F.avg_pool2d(patch_embeds, pool_size)
        patch_embeds = patch_embeds.view(bsz, dim, -1).permute(0, 2, 1)

        image_embeds = torch.cat([cls_embed, patch_embeds], dim=1)

        return image_embeds

    def forward(
        self,
        samples,
        max_output_length=10,
        is_train=True,
    ):
        """
        prompts: [before_vision, between_vision_and_text, after_text]
        """

        text_input = samples["text_input"]
        text_output = samples["text_output"]
        vision = samples["image"]  # will alter later to be compatible with videos

        if vision.ndim == 4:
            vision_embeds = self.encode_image(vision)
            vision_embeds = self.vision_to_text(vision_embeds)
        vision_atts = torch.ones(vision_embeds.size()[:-1], dtype=torch.long).to(
            self.device
        )

        text_before_vis = [self.prompts[0]] * len(text_input)
        text_after_vis = [self.prompts[1] + t for t in text_input]

        text_before_vis = self.tokenizer(text_before_vis, return_tensors="pt").to(
            self.device
        )
        text_after_vis = self.tokenizer(
            text_after_vis, return_tensors="pt", padding=True
        ).to(self.device)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            text_embeds_before_vis = self.text_model.shared(text_before_vis.input_ids)
            text_embeds_after_vis = self.text_model.shared(text_after_vis.input_ids)

            input_embeds = torch.cat(
                [text_embeds_before_vis, vision_embeds, text_embeds_after_vis], dim=1
            )
            input_atts = torch.cat(
                [
                    text_before_vis.attention_mask,
                    vision_atts,
                    text_after_vis.attention_mask,
                ],
                dim=1,
            )

            if is_train:
                text_output = self.tokenizer(
                    text_output, return_tensors="pt", padding=True
                ).to(self.device)
                targets = text_output.input_ids
                targets[targets == self.tokenizer.pad_token_id] = -100
                outputs = self.text_model(
                    inputs_embeds=input_embeds,
                    attention_mask=input_atts,
                    return_dict=True,
                    # decoder_input_ids = output_text.input_ids,
                    labels=targets,
                )
                return {"loss": outputs.loss}
            else:
                outputs = self.text_model.generate(
                    inputs_embeds=input_embeds,
                    attention_mask=input_atts,
                    # num_beams = 1,
                    max_length=max_output_length,
                )

                outputs = [
                    self.tokenizer.decode(output, skip_special_tokens=True)
                    for output in outputs
                ]

                return outputs

    def predict_answers(self, samples, max_len=10, **kwargs):
        return self.forward(samples, is_train=False, max_output_length=max_len)

    @classmethod
    def _build_from_cfg(cls, cfg=None):
        lm_type = cfg.get("lm_type", "bigscience/T0_3B")
        vit_type = cfg.get("vit_type", "base")
        prompts = cfg.get("prompts")

        valid_vit_type = ["base", "large"]
        assert (
            vit_type in valid_vit_type
        ), f"vit_type must be {valid_vit_type}, found {vit_type}."

        assert len(prompts) == 2, "prompts must be provided and contain two segments."

        model = cls(vit=vit_type, lm=lm_type, prompts=prompts)

        return model
