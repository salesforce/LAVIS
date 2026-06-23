import  os, shutil, subprocess
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast

from ldm.util import instantiate_from_config
from ldm.models.diffusion.plms import PLMSSampler

from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor

from app.utils import (
    get_pending_jobs,
    create_uniq_user_job_name
)

from app import job_output_path, finished_job_path

job_type = 'txt2image'
if torch.cuda.is_available():
    torch.cuda.set_device(1)
    device = "cuda"
else:
    device = "cpu"


# load safety model
safety_model_id = "CompVis/stable-diffusion-safety-checker"
safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images

def load_model_from_config(config, ckpt, verbose=False):
    #print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    if not os.path.exists('prompts'):
        os.makedirs('prompts', exist_ok=True)
    else:
        shutil.rmtree('prompts')
        os.makedirs('prompts', exist_ok=True)

    model.half()
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    return model


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x


def check_safety(x_image):
    safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
    x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
    assert x_checked_image.shape[0] == len(has_nsfw_concept)
    for i in range(len(has_nsfw_concept)):
        if has_nsfw_concept[i]:
            x_checked_image[i] = load_replacement(x_checked_image[i])
    return x_checked_image, has_nsfw_concept

def back_end():
    config = 'stable-diffusion/configs/stable-diffusion/v1-inference.yaml'
    ckpt = 'stable-diffusion/sd-v1-4.ckpt'
    config = OmegaConf.load(f"{config}")
    model = load_model_from_config(config, f"{ckpt}")
    print(device)
    model = model.to(device)
    outpath = os.path.join(job_output_path, job_type)
    sample_path = os.path.join(outpath, "samples")
    if not os.path.exists(sample_path):
        subprocess.run(['mkdir', '-p', sample_path], shell=False)
    finished_path = os.path.join(finished_job_path,job_type)
    if not os.path.exists(finished_path):
        subprocess.run(['mkdir', '-p',finished_path], shell=False)
    while True:
        pending_jobs = get_pending_jobs(job_type)
        for job in pending_jobs:
            while True:
                with open(job) as f:
                    content = f.readline().rstrip(' \n')
                    if len(content.split('\t')) == 4: break
            random_seed, time_stamp, user_prompt, num_images = content.split('\t')
            generate_image(model, int(random_seed), time_stamp, user_prompt, int(num_images), sample_path)
            shutil.move(job, finished_path)


def generate_image(model, random_seed, time_stamp, user_prompt, num_images, sample_path):
    scale = 7.5
    num_latent_channels = 4
    down_sample_factor = 8
    H, W = 512, 512
    ddim_steps = 50
    skip_grid = False
    skip_save = False
    n_rows = num_images
    precision_scope = autocast
    data = [[user_prompt] * num_images]
    seed_everything(random_seed)
    model = model.to(device)
    sampler = PLMSSampler(model)
    with torch.no_grad():
        with precision_scope(device):
        #with precision_scope("cuda"):
            with model.ema_scope():
                tic = time.time()
                all_samples = list()
                for n in trange(1, desc="Sampling"):
                    for prompts in tqdm(data, desc="data"):
                        uc = None
                        if scale != 1.0:
                            uc = model.get_learned_conditioning(num_images * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = model.get_learned_conditioning(prompts)
                        shape = [num_latent_channels, H // down_sample_factor, W // down_sample_factor]
                        samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                            conditioning=c,
                                                            batch_size= num_images,
                                                            shape=shape,
                                                            verbose=False,
                                                            unconditional_guidance_scale=scale,
                                                            unconditional_conditioning=uc,
                                                            eta=0.0,
                                                            x_T=None)

                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                        x_checked_image, has_nsfw_concept = check_safety(x_samples_ddim)

                        x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)

                        if not skip_save:
                            count = 1
                            for x_sample in x_checked_image_torch:
                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                img = Image.fromarray(x_sample.astype(np.uint8))
                                #img = put_watermark(img, wm_encoder)
                                img.save(os.path.join(sample_path, "{}_{}.png".format(create_uniq_user_job_name(str(time_stamp), user_prompt), count)))
                                count += 1
                        if not skip_grid:
                            all_samples.append(x_checked_image_torch)

                if not skip_grid:
                        # additionally, save as grid
                    grid = torch.stack(all_samples, 0)
                    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                    grid = make_grid(grid, nrow=n_rows)

                        # to image
                    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                    img = Image.fromarray(grid.astype(np.uint8))
                        #img = put_watermark(img, wm_encoder)
                    img.save(os.path.join(sample_path, "{}_grid.png".format(create_uniq_user_job_name(str(time_stamp), user_prompt))))
if __name__ == '__main__':
    back_end()
