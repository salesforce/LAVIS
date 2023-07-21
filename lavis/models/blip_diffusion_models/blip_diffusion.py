"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging
import os

import torch
import torch.nn.functional as F
import tqdm
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    DDIMScheduler,
    PNDMScheduler,
    UNet2DConditionModel,
)
from torch import nn
from transformers import CLIPTokenizer
from transformers.activations import QuickGELUActivation as QuickGELU

from lavis.common.registry import registry
from lavis.common.utils import download_and_untar, is_url
from lavis.models.base_model import BaseModel
from lavis.models.blip2_models.blip2_qformer import Blip2Qformer
from lavis.models.blip_diffusion_models.modeling_ctx_clip import CtxCLIPTextModel
from lavis.models.blip_diffusion_models.utils import numpy_to_pil, prepare_cond_image


class ProjLayer(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, drop_p=0.1, eps=1e-12):
        super().__init__()

        # Dense1 -> Act -> Dense2 -> Drop -> Res -> Norm
        self.dense1 = nn.Linear(in_dim, hidden_dim)
        self.act_fn = QuickGELU()
        self.dense2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(drop_p)

        self.LayerNorm = nn.LayerNorm(out_dim, eps=eps)

    def forward(self, x):
        x_in = x

        x = self.LayerNorm(x)
        x = self.dropout(self.dense2(self.act_fn(self.dense1(x)))) + x_in

        return x


@registry.register_model("blip_diffusion")
class BlipDiffusion(BaseModel):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "base": "configs/models/blip-diffusion/blip_diffusion_base.yaml",
        "canny": "configs/models/blip-diffusion/blip_diffusion_controlnet_canny.yaml",
        "depth": "configs/models/blip-diffusion/blip_diffusion_controlnet_depth.yaml",
        "hed": "configs/models/blip-diffusion/blip_diffusion_controlnet_hed.yaml",
    }

    def __init__(
        self,
        vit_model="clip_L",
        qformer_num_query_token=16,
        qformer_cross_attention_freq=1,
        qformer_pretrained_path=None,
        qformer_train=False,
        sd_pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5",
        sd_train_text_encoder=False,
        controlnet_pretrained_model_name_or_path=None,
        vae_half_precision=False,
        proj_train=False,
    ):
        super().__init__()

        self.num_query_token = qformer_num_query_token

        # BLIP-2
        self.blip = Blip2Qformer(
            vit_model=vit_model,
            num_query_token=qformer_num_query_token,
            cross_attention_freq=qformer_cross_attention_freq,
        )
        if qformer_pretrained_path is not None:
            state_dict = torch.load(qformer_pretrained_path, map_location="cpu")[
                "model"
            ]
            # qformer keys: Qformer.bert.encoder.layer.1.attention.self.key.weight
            # ckpt keys: text_model.bert.encoder.layer.1.attention.self.key.weight
            for k in list(state_dict.keys()):
                if "text_model" in k:
                    state_dict[k.replace("text_model", "Qformer")] = state_dict.pop(k)

            msg = self.blip.load_state_dict(state_dict, strict=False)
            assert all(["visual" in k for k in msg.missing_keys])
            assert len(msg.unexpected_keys) == 0

        self.qformer_train = qformer_train

        # projection layer
        self.proj_layer = ProjLayer(
            in_dim=768, out_dim=768, hidden_dim=3072, drop_p=0.1, eps=1e-12
        )
        self.proj_train = proj_train

        # stable diffusion
        self.tokenizer = CLIPTokenizer.from_pretrained(
            sd_pretrained_model_name_or_path, subfolder="tokenizer"
        )
        self.text_encoder = CtxCLIPTextModel.from_pretrained(
            sd_pretrained_model_name_or_path, subfolder="text_encoder"
        )
        self.vae = AutoencoderKL.from_pretrained(
            sd_pretrained_model_name_or_path, subfolder="vae"
        )
        if vae_half_precision:
            self.vae.half()

        self.unet = UNet2DConditionModel.from_pretrained(
            sd_pretrained_model_name_or_path, subfolder="unet"
        )
        # self.unet.enable_xformers_memory_efficient_attention()

        self.noise_scheduler = DDPMScheduler.from_config(
            sd_pretrained_model_name_or_path, subfolder="scheduler"
        )

        self.sd_train_text_encoder = sd_train_text_encoder

        if controlnet_pretrained_model_name_or_path is not None:
            self.controlnet = ControlNetModel.from_pretrained(
                controlnet_pretrained_model_name_or_path
            )

        self.freeze_modules()

        self.ctx_embeddings_cache = nn.Parameter(
            torch.zeros(1, self.num_query_token, 768), requires_grad=False
        )
        self._use_embeddings_cache = False

        # inference-related
        self._CTX_BEGIN_POS = 2

    def freeze_modules(self):
        to_freeze = [self.vae]
        if not self.sd_train_text_encoder:
            to_freeze.append(self.text_encoder)

        if not self.qformer_train:
            to_freeze.append(self.blip)

        if not self.proj_train:
            to_freeze.append(self.proj_layer)

        for module in to_freeze:
            module.eval()
            module.train = self.disabled_train
            module.requires_grad_(False)

    def disabled_train(self, mode=True):
        """Overwrite model.train with this function to make sure train/eval mode
        does not change anymore."""
        return self

    @property
    def pndm_scheduler(self):
        if not hasattr(self, "_pndm_scheduler"):
            self._pndm_scheduler = PNDMScheduler(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                set_alpha_to_one=False,
                skip_prk_steps=True,
            )
        return self._pndm_scheduler

    @property
    def ddim_scheduler(self):
        if not hasattr(self, "_ddim_scheduler"):
            self._ddim_scheduler = DDIMScheduler.from_config(
                "runwayml/stable-diffusion-v1-5", subfolder="scheduler"
            )
        return self._ddim_scheduler

    def _build_prompt(self, prompts, tgt_subjects, prompt_strength=1.0, prompt_reps=20):
        rv = []
        for prompt, tgt_subject in zip(prompts, tgt_subjects):
            prompt = f"a {tgt_subject} {prompt.strip()}"
            # a trick to amplify the prompt
            rv.append(", ".join([prompt] * int(prompt_strength * prompt_reps)))

        return rv

    def _predict_noise(
        self,
        t,
        latent_model_input,
        text_embeddings,
        width=512,
        height=512,
        cond_image=None,
    ):
        if hasattr(self, "controlnet"):
            cond_image = prepare_cond_image(
                cond_image, width, height, batch_size=1, device=self.device
            )

            down_block_res_samples, mid_block_res_sample = self.controlnet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embeddings,
                controlnet_cond=cond_image,
                # conditioning_scale=controlnet_condition_scale,
                return_dict=False,
            )
        else:
            down_block_res_samples, mid_block_res_sample = None, None

        noise_pred = self.unet(
            latent_model_input,
            timestep=t,
            encoder_hidden_states=text_embeddings,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
        )["sample"]

        return noise_pred

    def _init_latent(self, latent, height, width, generator, batch_size):
        if latent is None:
            latent = torch.randn(
                (1, self.unet.in_channels, height // 8, width // 8),
                generator=generator,
                device=generator.device,
            )
        latent = latent.expand(
            batch_size,
            self.unet.in_channels,
            height // 8,
            width // 8,
        )
        return latent.to(self.device)

    def _forward_prompt_embeddings(self, input_image, src_subject, prompt):
        # 1. extract BLIP query features and proj to text space -> (bs, 32, 768)
        query_embeds = self.forward_ctx_embeddings(input_image, src_subject)

        # 2. embeddings for prompt, with query_embeds as context
        tokenized_prompt = self._tokenize_text(prompt).to(self.device)
        text_embeddings = self.text_encoder(
            input_ids=tokenized_prompt.input_ids,
            ctx_embeddings=query_embeds,
            ctx_begin_pos=[self._CTX_BEGIN_POS],
        )[0]

        return text_embeddings

    @torch.no_grad()
    def get_image_latents(self, image, sample=True, rng_generator=None):
        assert isinstance(image, torch.Tensor)

        encoding_dist = self.vae.encode(image).latent_dist
        if sample:
            encoding = encoding_dist.sample(generator=rng_generator)
        else:
            encoding = encoding_dist.mode()
        latents = encoding * 0.18215
        return latents

    @torch.no_grad()
    def generate(
        self,
        samples,
        latents=None,
        guidance_scale=7.5,
        height=512,
        width=512,
        seed=42,
        num_inference_steps=50,
        neg_prompt="",
        controller=None,
        prompt_strength=1.0,
        prompt_reps=20,
        use_ddim=False,
    ):
        if controller is not None:
            self._register_attention_refine(controller)

        cond_image = samples["cond_images"]  # reference image
        cond_subject = samples["cond_subject"]  # source subject category
        tgt_subject = samples["tgt_subject"]  # target subject category
        prompt = samples["prompt"]
        cldm_cond_image = samples.get("cldm_cond_image", None)  # conditional image

        prompt = self._build_prompt(
            prompts=prompt,
            tgt_subjects=tgt_subject,
            prompt_strength=prompt_strength,
            prompt_reps=prompt_reps,
        )

        text_embeddings = self._forward_prompt_embeddings(
            cond_image, cond_subject, prompt
        )

        # 3. unconditional embedding
        do_classifier_free_guidance = guidance_scale > 1.0
        if do_classifier_free_guidance:
            max_length = self.text_encoder.text_model.config.max_position_embeddings

            uncond_input = self.tokenizer(
                [neg_prompt],
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
            uncond_embeddings = self.text_encoder(
                input_ids=uncond_input.input_ids.to(self.device),
                ctx_embeddings=None,
            )[0]

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        if seed is not None:
            generator = torch.Generator(device=self.device)
            generator = generator.manual_seed(seed)

        latents = self._init_latent(latents, height, width, generator, batch_size=1)

        scheduler = self.pndm_scheduler if not use_ddim else self.ddim_scheduler

        # set timesteps
        extra_set_kwargs = {}
        scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

        iterator = tqdm.tqdm(scheduler.timesteps)

        for i, t in enumerate(iterator):
            latents = self._denoise_latent_step(
                latents=latents,
                t=t,
                text_embeddings=text_embeddings,
                cond_image=cldm_cond_image,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                use_inversion=use_ddim,
            )

        image = self._latent_to_image(latents)

        return image

    def _latent_to_image(self, latents):
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()

        image = numpy_to_pil(image)

        return image

    def _denoise_latent_step(
        self,
        latents,
        t,
        text_embeddings,
        guidance_scale,
        height,
        width,
        cond_image=None,
        use_inversion=False,
    ):
        if use_inversion:
            noise_placeholder = []

        # expand the latents if we are doing classifier free guidance
        do_classifier_free_guidance = guidance_scale > 1.0

        latent_model_input = (
            torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        )

        # predict the noise residual
        noise_pred = self._predict_noise(
            t=t,
            latent_model_input=latent_model_input,
            text_embeddings=text_embeddings,
            width=width,
            height=height,
            cond_image=cond_image,
        )

        if use_inversion:
            noise_placeholder.append(noise_pred[2].unsqueeze(0))

        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

        if use_inversion:
            noise_placeholder.append(noise_pred[-1].unsqueeze(0))
            noise_pred = torch.cat(noise_placeholder)

        # compute the previous noisy sample x_t -> x_t-1
        scheduler = self.ddim_scheduler if use_inversion else self.pndm_scheduler

        latents = scheduler.step(
            noise_pred,
            t,
            latents,
        )["prev_sample"]

        return latents

    def _tokenize_text(self, text_input, with_query=True):
        max_len = self.text_encoder.text_model.config.max_position_embeddings
        if with_query:
            max_len -= self.num_query_token

        tokenized_text = self.tokenizer(
            text_input,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )

        return tokenized_text

    def forward_ctx_embeddings(self, input_image, text_input, ratio=None):
        def compute_ctx_embeddings(input_image, text_input):
            # blip_embeddings = self.blip(image=input_image, text=text_input)
            blip_embeddings = self.blip.extract_features(
                {"image": input_image, "text_input": text_input}, mode="multimodal"
            ).multimodal_embeds
            ctx_embeddings = self.proj_layer(blip_embeddings)

            return ctx_embeddings

        if isinstance(text_input, str):
            text_input = [text_input]

        if self._use_embeddings_cache:
            # expand to batch size
            ctx_embeddings = self.ctx_embeddings_cache.expand(len(text_input), -1, -1)
        else:
            if isinstance(text_input[0], str):
                text_input, input_image = [text_input], [input_image]

            all_ctx_embeddings = []

            for inp_image, inp_text in zip(input_image, text_input):
                ctx_embeddings = compute_ctx_embeddings(inp_image, inp_text)
                all_ctx_embeddings.append(ctx_embeddings)

            if ratio is not None:
                assert len(ratio) == len(all_ctx_embeddings)
                assert sum(ratio) == 1
            else:
                ratio = [1 / len(all_ctx_embeddings)] * len(all_ctx_embeddings)

            ctx_embeddings = torch.zeros_like(all_ctx_embeddings[0])

            for ratio, ctx_embeddings_ in zip(ratio, all_ctx_embeddings):
                ctx_embeddings += ratio * ctx_embeddings_

        return ctx_embeddings

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "clip_L")

        qformer_cross_attention_freq = cfg.get("qformer_cross_attention_freq", 1)
        qformer_num_query_token = cfg.get("qformer_num_query_token", 16)
        qformer_train = cfg.get("qformer_train", False)

        sd_train_text_encoder = cfg.get("sd_train_text_encoder", False)
        sd_pretrained_model_name_or_path = cfg.get(
            "sd_pretrained_model_name_or_path", "runwayml/stable-diffusion-v1-5"
        )

        controlnet_pretrained_model_name_or_path = cfg.get(
            "controlnet_pretrained_model_name_or_path", None
        )

        vae_half_precision = cfg.get("vae_half_precision", False)

        model = cls(
            vit_model=vit_model,
            qformer_cross_attention_freq=qformer_cross_attention_freq,
            qformer_num_query_token=qformer_num_query_token,
            qformer_train=qformer_train,
            sd_train_text_encoder=sd_train_text_encoder,
            sd_pretrained_model_name_or_path=sd_pretrained_model_name_or_path,
            controlnet_pretrained_model_name_or_path=controlnet_pretrained_model_name_or_path,
            vae_half_precision=vae_half_precision,
        )
        model.load_checkpoint_from_config(cfg)

        return model

    def load_checkpoint_from_dir(self, checkpoint_dir_or_url):
        # if checkpoint_dir is a url, download it and untar it
        if is_url(checkpoint_dir_or_url):
            checkpoint_dir_or_url = download_and_untar(checkpoint_dir_or_url)

        logging.info(f"Loading pretrained model from {checkpoint_dir_or_url}")

        def load_state_dict(module, filename):
            try:
                state_dict = torch.load(
                    os.path.join(checkpoint_dir_or_url, filename), map_location="cpu"
                )
                msg = module.load_state_dict(state_dict, strict=False)
            except FileNotFoundError:
                logging.info("File not found, skip loading: {}".format(filename))

        load_state_dict(self.proj_layer, "proj_layer/proj_weight.pt")
        load_state_dict(self.blip, "blip_model/blip_weight.pt")
        load_state_dict(self.unet, "unet/diffusion_pytorch_model.bin")
        load_state_dict(self.vae, "vae/diffusion_pytorch_model.bin")
        load_state_dict(self.text_encoder, "text_encoder/pytorch_model.bin")

        try:
            self.ctx_embeddings_cache.data = torch.load(
                os.path.join(
                    checkpoint_dir_or_url, "ctx_embeddings_cache/ctx_embeddings_cache.pt"
                ),
                map_location=self.device,
            )
            self._use_embeddings_cache = True
            print("Loaded ctx_embeddings_cache from {}".format(checkpoint_dir_or_url))
        except FileNotFoundError:
            self._use_embeddings_cache = False
            print("No ctx_embeddings_cache found in {}".format(checkpoint_dir_or_url))

    def load_from_pretrained(self, url_or_filename):
        checkpoint_dir = url_or_filename
        self.load_checkpoint_from_dir(checkpoint_dir)

    def load_checkpoint(self, url_or_filename):
        """
        Used to load finetuned models.
        """

        super().load_checkpoint(url_or_filename)

        print("loading fine-tuned model from {}".format(url_or_filename))
        self._use_embeddings_cache = True
