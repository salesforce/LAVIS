"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import numpy as np
import torch
from diffusers import ControlNetModel
from diffusers.utils.pil_utils import PIL_INTERPOLATION
from PIL import Image

from lavis.common.registry import registry
from lavis.models.blip_diffusion_models.blip_diffusion import BlipDiffusion


@registry.register_model("blip_diffusion_controlnet")
class BlipDiffusionControlNet(BlipDiffusion):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "canny": "configs/models/blip-diffusion/blip_diffusion_controlnet_canny.yaml",
        "depth": "configs/models/blip-diffusion/blip_diffusion_controlnet_depth.yaml",
        "hed": "configs/models/blip-diffusion/blip_diffusion_controlnet_hed.yaml",
    }

    def __init__(
        self,
        vit_model="clip_L",
        qformer_num_query_token=16,
        qformer_cross_attention_freq=1,
        qformer_pretrained_path="/export/share/junnan-li/BLIP2/checkpoint/clip_q16.pth",
        sd_pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5",
        sd_train_text_encoder=False,
        controlnet_pretrained_model_name_or_path=None,
    ):
        super().__init__(
            vit_model=vit_model,
            qformer_num_query_token=qformer_num_query_token,
            qformer_cross_attention_freq=qformer_cross_attention_freq,
            qformer_pretrained_path=qformer_pretrained_path,
            sd_pretrained_model_name_or_path=sd_pretrained_model_name_or_path,
            sd_train_text_encoder=sd_train_text_encoder,
        )

        # controlnet
        assert (
            controlnet_pretrained_model_name_or_path is not None
        ), "controlnet_pretrained_model_name_or_path must be specified"
        self.controlnet = ControlNetModel.from_pretrained(
            controlnet_pretrained_model_name_or_path
        )

    def forward(self, **kwargs):
        raise NotImplementedError

    def prepare_cond_image(
        self, image, width, height, batch_size, do_classifier_free_guidance=True
    ):
        if not isinstance(image, torch.Tensor):
            if isinstance(image, Image.Image):
                image = [image]

            if isinstance(image[0], Image.Image):
                images = []

                for image_ in image:
                    image_ = image_.convert("RGB")
                    image_ = image_.resize(
                        (width, height), resample=PIL_INTERPOLATION["lanczos"]
                    )
                    image_ = np.array(image_)
                    image_ = image_[None, :]
                    images.append(image_)

                image = images

                image = np.concatenate(image, axis=0)
                image = np.array(image).astype(np.float32) / 255.0
                image = image.transpose(0, 3, 1, 2)
                image = torch.from_numpy(image)
            elif isinstance(image[0], torch.Tensor):
                image = torch.cat(image, dim=0)

        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            # repeat_by = num_images_per_prompt
            raise NotImplementedError

        image = image.repeat_interleave(repeat_by, dim=0)

        # image = image.to(device=self.device, dtype=dtype)
        image = image.to(device=self.device)

        if do_classifier_free_guidance:
            image = torch.cat([image] * 2)

        return image

    def _predict_noise(
        self, samples, t, latent_model_input, text_embeddings, width, height
    ):
        # [TODO] support batched inference
        assert "cond_image" in samples, "cond_image must be provided in samples."
        cond_image = samples["cond_image"]  # condition image for controlnet
        cond_image = self.prepare_cond_image(cond_image, width, height, batch_size=1)

        down_block_res_samples, mid_block_res_sample = self.controlnet(
            latent_model_input,
            t,
            encoder_hidden_states=text_embeddings,
            controlnet_cond=cond_image,
            # conditioning_scale=controlnet_condition_scale,
            return_dict=False,
        )

        # predict the noise residual
        noise_pred = self.unet(
            latent_model_input,
            timestep=t,
            encoder_hidden_states=text_embeddings,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
        )["sample"]

        return noise_pred

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "clip_L")
        cross_attention_freq = cfg.get("cross_attention_freq", 1)
        num_query_token = cfg.get("num_query_token", 16)

        sd_train_text_encoder = cfg.get("sd_train_text_encoder", False)
        sd_pretrained_model_name_or_path = cfg.get(
            "sd_pretrained_model_name_or_path", "runwayml/stable-diffusion-v1-5"
        )

        controlnet_pretrained_model_name_or_path = cfg.get(
            "controlnet_pretrained_model_name_or_path", None
        )

        model = cls(
            vit_model=vit_model,
            qformer_cross_attention_freq=cross_attention_freq,
            qformer_num_query_token=num_query_token,
            sd_train_text_encoder=sd_train_text_encoder,
            sd_pretrained_model_name_or_path=sd_pretrained_model_name_or_path,
            controlnet_pretrained_model_name_or_path=controlnet_pretrained_model_name_or_path,
        )
        model.load_checkpoint_from_config(cfg)

        return model
