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
  - zero-shot inference: [notebook](https://github.com/dxli94/LAVIS-2/tree/20230623-blip-diffusion-documentation/projects/blip-diffusion/notebooks/generation_zeroshot.ipynb)
  - inference with fine-tuned checkpoint: [notebook](https://github.com/dxli94/LAVIS-2/blob/main/projects/blip-diffusion/notebooks/generation_finetuned_dog.ipynb)

- **Structure-Controlled Generation / Stylization**: [notebook](https://github.com/dxli94/LAVIS-2/tree/20230623-blip-diffusion-documentation/projects/blip-diffusion/notebooks/stylization.ipynb)

- **Subject-driven Editing**:
  - editing a synthetic image:
    - First generate an image, then edit the image with the specified subject visuals: [notebook](https://github.com/dxli94/LAVIS-2/tree/20230623-blip-diffusion-documentation/projects/blip-diffusion/notebooks/editing_synthetic_zeroshot.ipynb)
  - editing a real image with DDIM inversion:
    - zero-shot inference: [notebook](https://github.com/dxli94/LAVIS-2/tree/20230623-blip-diffusion-documentation/projects/blip-diffusion/notebooks/editing_real_zeroshot.ipynb)
    - inference with fine-tuned checkpoint: [notebook](https://github.com/dxli94/LAVIS-2/blob/main/projects/blip-diffusion/notebooks/editing_real_finetuned.ipynb)

- **Virtual Try-On via Subject-driven Editing**:
  - the model can be used to naturally facilitate virtual try-on. We provide an zero-shot example: [notebook](https://github.com/dxli94/LAVIS-2/tree/20230623-blip-diffusion-documentation/projects/blip-diffusion/notebooks/editing_tryon_zeroshot.ipynb);
  - try fine-tuning the model for better subject fidelity.


### Fine-tuning
1. Prepare example images of a subject and put it in a folder, e.g. ``image_path``
2. Open ``run_scripts/blip-diffusion/train_db.sh``:
-  change ``IMAGE_STORAGE`` to point to ``image_path``
-  change ``SUBJECT_TEXT`` accordingly.
-  change other training hparams as wish.

Finally,

```bash
bash run_scripts/blip-diffusion/train_db.sh
```

**Tips on fine-tuning**:

1. For common subject classes, animals (dogs, cats), vehicles, etc, we find 30-50 steps sufficient;
2. for highly-customized subjects, 80-120 steps are usually needed;
3. overly many fine-tuning steps or high learning rates usually lead to model overfitting to the input example images and failing to address text prompts in generaitons;
3. more subject examples with higher diveristy in terms of environment/style gives better fine-tuning results. We use 3-5 images most of the time when reporting results in the paper, yet more is better.
4. For editing-related applications, overfitting to the input images may not be an issue. For example, in the virtual try-on example, we use a single image for 120 fine-tuning steps.


### Citing BLIP-Diffusion
<pre>
@article{li2023blip,
  title={BLIP-Diffusion: Pre-trained Subject Representation for Controllable Text-to-Image Generation and Editing},
  author={Li, Dongxu and Li, Junnan and Hoi, Steven CH},
  journal={arXiv preprint arXiv:2305.14720},
  year={2023}
}
</pre>