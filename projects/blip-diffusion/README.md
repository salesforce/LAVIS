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
  - zero-shot inference: [notebook](https://github.com/salesforce/LAVIS/blob/main/projects/blip-diffusion/notebooks/generation_zeroshot.ipynb)
  - inference with fine-tuned checkpoint: [notebook](https://github.com/salesforce/LAVIS/blob/main/projects/blip-diffusion/notebooks/generation_finetuned_dog.ipynb)

- **Structure-Controlled Generation / Stylization**: [notebook](https://github.com/salesforce/LAVIS/blob/main/projects/blip-diffusion/notebooks/stylization.ipynb)

### Cite BLIP-Diffusion
If you find our work helpful, please consider citing:
<pre>
@article{li2023blip,
  title={BLIP-Diffusion: Pre-trained Subject Representation for Controllable Text-to-Image Generation and Editing},
  author={Li, Dongxu and Li, Junnan and Hoi, Steven CH},
  journal={arXiv preprint arXiv:2305.14720},
  year={2023}
}
</pre>
