## BLIP-Diffusion: Pre-trained Subject Representation for Controllable Text-to-Image Generation and Editing
[Paper](https://arxiv.org/abs/2305.14720), [Demo Site](https://dxli94.github.io/BLIP-Diffusion-website/), [Video](https://youtu.be/Wf09s4JnDb0)

This repo hosts the official implementation of BLIP-Diffusion, a text-to-image diffusion model with built-in support for multimodal subject-and-text condition. BLIP-Diffusion enables zero-shot subject-driven generation, and efficient fine-tuning for customized subjects with up to 20x speedup. In addition, BLIP-Diffusion can be flexibly combiend with ControlNet and prompt-to-prompt to enable novel subject-driven generation and editing applications.

<img src="teaser-website.png" width="800">


### Installation

Install the LAVIS library from source:

```bash
pip install -e .
```

### Notebook Examples
- **Subject-driven Generation**: 
  - zero-shot inference: [notebook](https://github.com/salesforce/LAVIS/blob/main/projects/blip-diffusion/notebooks/generation_zeroshot.ipynb), [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/salesforce/LAVIS/blob/main/projects/blip-diffusion/notebooks/generation_zeroshot.ipynb)
  - inference with fine-tuned checkpoint: [notebook](https://github.com/salesforce/LAVIS/blob/main/projects/blip-diffusion/notebooks/generation_finetuned_dog.ipynb), [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/salesforce/LAVIS/blob/main/projects/blip-diffusion/notebooks/generation_finetuned_dog.ipynb)

- **Structure-Controlled Generation / Stylization**: [notebook](https://github.com/salesforce/LAVIS/blob/main/projects/blip-diffusion/notebooks/stylization.ipynb), [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/salesforce/LAVIS/blob/main/projects/blip-diffusion/notebooks/stylization.ipynb)

- **Subject-driven Editing**:
  - editing a synthetic image:
    - First generate an image, then edit the image with the specified subject visuals: [notebook](https://github.com/salesforce/LAVIS/blob/main/projects/blip-diffusion/notebooks/editing_synthetic_zeroshot.ipynb), [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/salesforce/LAVIS/blob/main/projects/blip-diffusion/notebooks/editing_synthetic_zeroshot.ipynb) 
  - editing a real image with DDIM inversion:
    - zero-shot inference: [notebook](https://github.com/salesforce/LAVIS/blob/main/projects/blip-diffusion/notebooks/editing_real_zeroshot.ipynb), [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/salesforce/LAVIS/blob/main/projects/blip-diffusion/notebooks/editing_real_zeroshot.ipynb)
    - inference with fine-tuned checkpoint: [notebook](https://github.com/salesforce/LAVIS/blob/main/projects/blip-diffusion/notebooks/editing_real_finetuned.ipynb), [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/salesforce/LAVIS/blob/main/projects/blip-diffusion/notebooks/editing_real_finetuned.ipynb)

- **Virtual Try-On via Subject-driven Editing**:
  - the model can be used to naturally facilitate virtual try-on. We provide an zero-shot example: [notebook](https://github.com/salesforce/LAVIS/blob/main/projects/blip-diffusion/notebooks/editing_tryon_zeroshot.ipynb), [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/salesforce/LAVIS/blob/main/projects/blip-diffusion/notebooks/editing_tryon_zeroshot.ipynb);

### **🧨 Diffusers Support**
BLIP-Diffusion is now supported in 🧨[Diffusers](https://huggingface.co/docs/diffusers/main/en/api/pipelines/blip_diffusion).
- Example on subject-driven generation:
```python
from diffusers.pipelines import BlipDiffusionPipeline
from diffusers.utils import load_image
import torch

blip_diffusion_pipe = BlipDiffusionPipeline.from_pretrained(
    "Salesforce/blipdiffusion", torch_dtype=torch.float16
).to("cuda")


cond_subject = "dog"
tgt_subject = "dog"
text_prompt_input = "swimming underwater"

cond_image = load_image(
    "https://huggingface.co/datasets/ayushtues/blipdiffusion_images/resolve/main/dog.jpg"
)
guidance_scale = 7.5
num_inference_steps = 25
negative_prompt = "over-exposure, under-exposure, saturated, duplicate, out of frame, lowres, cropped, worst quality, low quality, jpeg artifacts, morbid, mutilated, out of frame, ugly, bad anatomy, bad proportions, deformed, blurry, duplicate"


output = blip_diffusion_pipe(
    text_prompt_input,
    cond_image,
    cond_subject,
    tgt_subject,
    guidance_scale=guidance_scale,
    num_inference_steps=num_inference_steps,
    neg_prompt=negative_prompt,
    height=512,
    width=512,
).images
output[0].save("image.png")
```
- Example on subject-driven stylization
```python
from diffusers.pipelines import BlipDiffusionControlNetPipeline
from diffusers.utils import load_image
from controlnet_aux import CannyDetector
import torch

blip_diffusion_pipe = BlipDiffusionControlNetPipeline.from_pretrained(
    "Salesforce/blipdiffusion-controlnet", torch_dtype=torch.float16
).to("cuda")

style_subject = "flower"
tgt_subject = "teapot"
text_prompt = "on a marble table"

cldm_cond_image = load_image(
    "https://huggingface.co/datasets/ayushtues/blipdiffusion_images/resolve/main/kettle.jpg"
).resize((512, 512))
canny = CannyDetector()
cldm_cond_image = canny(cldm_cond_image, 30, 70, output_type="pil")
style_image = load_image(
    "https://huggingface.co/datasets/ayushtues/blipdiffusion_images/resolve/main/flower.jpg"
)
guidance_scale = 7.5
num_inference_steps = 50
negative_prompt = "over-exposure, under-exposure, saturated, duplicate, out of frame, lowres, cropped, worst quality, low quality, jpeg artifacts, morbid, mutilated, out of frame, ugly, bad anatomy, bad proportions, deformed, blurry, duplicate"


output = blip_diffusion_pipe(
    text_prompt,
    style_image,
    cldm_cond_image,
    style_subject,
    tgt_subject,
    guidance_scale=guidance_scale,
    num_inference_steps=num_inference_steps,
    neg_prompt=negative_prompt,
    height=512,
    width=512,
).images
output[0].save("image.png")
```


### Cite BLIP-Diffusion
If you find our work helpful, please consider citing:
<pre>
@article{li2023blip,
  title={BLIP-Diffusion: Pre-trained Subject Representation for Controllable Text-to-Image Generation and Editing},
  author={Li, Dongxu and Li, Junnan and Hoi, Steven CH},
  journal={arXiv preprint arXiv:2305.14720},
  year={2023}
}

@inproceedings{li2023lavis,
  title={LAVIS: A One-stop Library for Language-Vision Intelligence},
  author={Li, Dongxu and Li, Junnan and Le, Hung and Wang, Guangsen and Savarese, Silvio and Hoi, Steven CH},
  booktitle={Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations)},
  pages={31--41},
  year={2023}
}
</pre>
