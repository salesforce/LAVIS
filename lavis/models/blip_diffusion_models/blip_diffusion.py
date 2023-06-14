"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import inspect
import logging
import os

import torch
import tqdm
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    UNet2DConditionModel,
)
from torch import nn
from transformers import CLIPTokenizer
from transformers.activations import QuickGELUActivation as QuickGELU

from lavis.common.registry import registry
from lavis.models.base_model import BaseModel
from lavis.models.blip2_models.blip2_qformer import Blip2Qformer
from lavis.models.blip_diffusion_models.modeling_ctx_clip import CtxCLIPTextModel
from lavis.models.blip_diffusion_models.utils import numpy_to_pil


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
        "base": "configs/models/blip-diffusion/blip_diffusion_base.yaml"
    }

    def __init__(
        self,
        vit_model="clip_L",
        qformer_num_query_token=16,
        qformer_cross_attention_freq=1,
        qformer_pretrained_path="/export/share/junnan-li/BLIP2/checkpoint/clip_q16.pth",
        sd_pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5",
        sd_train_text_encoder=False,
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

        # projection layer
        self.proj_layer = ProjLayer(
            in_dim=768, out_dim=768, hidden_dim=3072, drop_p=0.1, eps=1e-12
        )

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
        self.unet = UNet2DConditionModel.from_pretrained(
            sd_pretrained_model_name_or_path, subfolder="unet"
        )
        self.noise_scheduler = DDPMScheduler.from_config(
            sd_pretrained_model_name_or_path, subfolder="scheduler"
        )

        self.sd_train_text_encoder = sd_train_text_encoder
        self.freeze_modules()

        self.ctx_embeddings_cache = None
        self.use_embeddings_cache = False

        # inference-related
        self._PROMPT_REPS = 20
        self._CTX_BEGIN_POS = 2

    def freeze_modules(self):
        to_freeze = [self.vae, self.unet]
        if not self.sd_train_text_encoder:
            to_freeze.append(self.text_encoder)

        for module in to_freeze:
            module.eval()
            module.train = self.disabled_train
            module.requires_grad_(False)

    def disabled_train(self, mode=True):
        """Overwrite model.train with this function to make sure train/eval mode
        does not change anymore."""
        return self

    @property
    def eval_noise_scheduler(self):
        if not hasattr(self, "_eval_noise_scheduler"):
            self._eval_noise_scheduler = PNDMScheduler(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                set_alpha_to_one=False,
                skip_prk_steps=True,
            )
        return self._eval_noise_scheduler

    def _build_prompt(self, prompts, tgt_subjects, prompt_strength=1.0):
        rv = []
        for prompt, tgt_subject in zip(prompts, tgt_subjects):
            prompt = f"a {tgt_subject} {prompt.strip()}"
            # a trick to amplify the prompt
            rv.append(", ".join([prompt] * int(prompt_strength * self._PROMPT_REPS)))

        return rv

    def _predict_noise(
        self, samples, t, latent_model_input, text_embeddings, width=512, height=512
    ):
        noise_pred = self.unet(
            latent_model_input, timestep=t, encoder_hidden_states=text_embeddings
        )["sample"]

        return noise_pred

    @torch.no_grad()
    def generate(
        self,
        samples,
        guidance_scale=7.5,
        height=512,
        width=512,
        seed=42,
        num_inference_steps=50,
        eta=1,
        neg_prompt="",
        controller=None,
        prompt_strength=1.0,
    ):
        # [TODO] support batched generation
        if controller is not None:
            self.register_attention_control(controller)

        input_image = samples["input_images"]  # reference image
        src_subject = samples["src_subject"]  # source subject category
        tgt_subject = samples["tgt_subject"]  # target subject category

        prompt = self._build_prompt(
            prompts=samples["prompt"],
            tgt_subjects=tgt_subject,
            prompt_strength=prompt_strength,
        )

        # 1. extract BLIP query features and proj to text space -> (bs, 32, 768)
        query_embeds = self.forward_ctx_embeddings(input_image, src_subject)

        # 2. embeddings for prompt, with query_embeds as context
        tokenized_prompt = self.tokenize_text(prompt).to(self.device)
        text_embeddings = self.text_encoder(
            input_ids=tokenized_prompt.input_ids,
            ctx_embeddings=query_embeds,
            ctx_begin_pos=[self._CTX_BEGIN_POS],
        )[0]

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

        latents_shape = (1, self.unet.in_channels, height // 8, width // 8)

        if seed is not None:
            generator = torch.Generator(device=self.device)
            generator = generator.manual_seed(seed)

        latents = torch.randn(
            latents_shape,
            generator=generator,
            device=self.device,
        )

        scheduler = self.eval_noise_scheduler

        # set timesteps
        accepts_offset = "offset" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        extra_set_kwargs = {}
        if accepts_offset:
            extra_set_kwargs["offset"] = 1

        scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

        # if we use LMSDiscreteScheduler, let's make sure latents are mulitplied by sigmas
        if isinstance(scheduler, LMSDiscreteScheduler):
            latents = latents * scheduler.sigmas[0]

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        iterator = tqdm.tqdm(scheduler.timesteps)

        for i, t in enumerate(iterator):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = (
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            )

            # predict the noise residual
            noise_pred = self._predict_noise(
                samples=samples,
                t=t,
                latent_model_input=latent_model_input,
                text_embeddings=text_embeddings,
                width=width,
                height=height,
            )

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred, t, latents, **extra_step_kwargs)[
                "prev_sample"
            ]

        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        image = numpy_to_pil(image)

        return image

    def tokenize_text(self, text_input, with_query=True):
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

        if self.ctx_embeddings_cache is not None and self.use_embeddings_cache:
            print("Using cached BLIP embeddings")
            # expand to batch size
            ctx_embeddings = self.ctx_embeddings_cache.expand(len(text_input), -1, -1)
        else:
            print("Computing BLIP embeddings for {} subjects".format(len(text_input)))
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

        sd_train_text_encoder = cfg.get("sd_train_text_encoder", False)
        sd_pretrained_model_name_or_path = cfg.get(
            "sd_pretrained_model_name_or_path", "runwayml/stable-diffusion-v1-5"
        )

        model = cls(
            vit_model=vit_model,
            qformer_cross_attention_freq=qformer_cross_attention_freq,
            qformer_num_query_token=qformer_num_query_token,
            sd_train_text_encoder=sd_train_text_encoder,
            sd_pretrained_model_name_or_path=sd_pretrained_model_name_or_path,
        )
        model.load_checkpoint_from_config(cfg)

        return model

    def load_checkpoint_from_dir(self, checkpoint_dir):
        logging.info(f"Loading pretrained model from {checkpoint_dir}")

        def load_state_dict(module, filename):
            state_dict = torch.load(
                os.path.join(checkpoint_dir, filename), map_location="cpu"
            )
            msg = module.load_state_dict(state_dict, strict=False)
            print(msg)

        load_state_dict(self.proj_layer, "proj_layer/proj_weight.pt")
        load_state_dict(self.blip, "blip_model/blip_weight.pt")
        load_state_dict(self.unet, "unet/diffusion_pytorch_model.bin")
        load_state_dict(self.vae, "vae/diffusion_pytorch_model.bin")
        load_state_dict(self.text_encoder, "text_encoder/pytorch_model.bin")

        try:
            self.ctx_embeddings_cache.data = torch.load(
                os.path.join(
                    checkpoint_dir, "ctx_embeddings_cache/ctx_embeddings_cache.pt"
                ),
                map_location="cpu",
            )
            self.use_embeddings_cache = True
            print("Loaded ctx_embeddings_cache from {}".format(checkpoint_dir))
        except FileNotFoundError:
            self.use_embeddings_cache = False
            print("No ctx_embeddings_cache found in {}".format(checkpoint_dir))

    def load_from_pretrained(self, url_or_filename):
        assert os.path.isdir(url_or_filename), "Must be a valid directory"

        checkpoint_dir = url_or_filename
        self.load_checkpoint_from_dir(checkpoint_dir)
