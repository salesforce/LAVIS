<p align="center">
    <br>
    <img src="docs/_statics/logo_final.png" width="400"/>
    <br>
<p>

# LAVIS - A Library for Language-Vision Intelligence

## Table of Contents
1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Installation](#installation)
4. [Documentation]()
5. [GUI Demo]()
<!-- 7. [How to Contribute](https://opensource.salesforce.com/OmniXAI/latest/omnixai.html#how-to-contribute) -->
<!-- 8. [Technical Report and Citing OmniXAI](#technical-report-and-citing-omnixai) -->

## Introduction
LAVIS is a Python deep learning library for LAnguage-and-VISion research and applications.
It features a unified design to access
- **10+** common tasks
(retrieval, captioning, visual question answering, multimodal classification etc.);
- **20+** datasets (COCO, Flickr, Nocaps, Conceptual
Commons, SBU, etc.);
- **30+** pretrained weights of state-of-the-art foundation language-vision models and their task-specific adaptations, including [ALBEF](https://arxiv.org/pdf/2107.07651.pdf),
[BLIP](https://arxiv.org/pdf/2201.12086.pdf), [ALPRO](https://arxiv.org/pdf/2112.09583.pdf), [CLIP](https://arxiv.org/pdf/2103.00020.pdf).


This library aims to provide engineers and researchers with a one-stop solution to rapidly develop models for their specific multimodal
scenarios, and benchmark them across standard and customized datasets.

Key features of LAVIS include:

- **Modular and Extensible Library Design**: facilitating to easily utilize and repurpose existing modules (datasets, models, preprocessors), also to add new modules.

- **Easy Off-the-shelf Inference and Feature Extraction**: readily available pre-trained models let you take advantage of state-of-the-art multimodal understanding and generation capabilities on your own data.

- **Reproducible Model Zoo**: provided training/pre-training recipies to easily replicate and extend state-of-the-art models.

- **Dataset Zoo and Automatic Downloading Tools**: it can be a hassle to prepare the many language-vision datasets. LAVIS provides automatic downloaing scripts to help prepare a large variety of datasets and their annotations.


The following table shows the supported tasks, datasets and models in our library. This is a continuing effort and we are working on further growing the list.

|                  Tasks                   |     Supported Models     |             Supported Datasets             | Modalities  |
| :--------------------------------------: | :----------------------: | :----------------------------------------: | :---------: |
|         Image-text Pre-training          |       ALBEF, BLIP        | COCO, VisualGenome, SBU ConceptualCaptions | image, text |
|           Image-text Retrieval           |    ALBEF, BLIP, CLIP     |              COCO, Flickr30k               | image, text |
|           Text-image Retrieval           |    ALBEF, BLIP, CLIP     |              COCO, Flickr30k               | image, text |
|        Visual Question Answering         |       ALBEF, BLIP        |           VQAv2, OKVQA, A-OKVQA            | image, text |
|             Image Captioning             |           BLIP           |                COCO, NoCaps                | image, text |
|           Image Classification           |           CLIP           |                  ImageNet                  | image, text |
| Natural Language Visual Reasoning (NLVR) |       ALBEF, BLIP        |                   NLVR2                    | image, text |
|          Visual Entailment (VE)          |          ALBEF           |                  SNLI-VE                   | image, text |
|             Visual Dialogue              |           BLIP           |                  VisDial                   | image, text |
|           Video-text Retrieval           |       BLIP, ALPRO        |               MSRVTT, DiDeMo               | video, text |
|           Text-video Retrieval           |       BLIP, ALPRO        |               MSRVTT, DiDeMo               | video, text |
|    Video Question Answering (VideoQA)    |       BLIP, ALPRO        |                MSRVTT, MSVD                | video, text |
|              Video Dialogue              |                          |                                            | video, text |
|      Multimodal Feature Extraction       | ALBEF, CLIP, BLIP, ALPRO |                 customized                 | image, text |

## Getting Started

### Image Captioning
*******************************
We now use the BLIP model to generate a caption for the image. To make inference even easier, we also associate each
pre-trained model with its preprocessors (transforms),  we use ``load_model_and_preprocess()`` with the following arguments:

- ``name``: The name of the model to load. This could be a pre-trained model, task model, or feature extractor. See ``model_zoo`` for a full list of model names.
- ``model_type``: Each architecture has variants trained on different datasets and at different scale. See Types column in ``model_zoo`` for a full list of model types.
- ``is_eval``: if `True`, set the model to evaluation mode. This is desired for inference or feature extraction.
- ``devce``: device to load the model to.

```python
from lavis.models import load_model_and_preprocess
# loads BLIP caption base model, with finetuned checkpoints on MSCOCO captioning dataset.
# this also loads the associated image processors
model, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=device)

# preprocess the image
# vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)
image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

# generate caption
model.generate({"image": image}
# ['a large fountain spewing water into the air']
```

You may also load models and their preprocessors separately via ``load_model()`` and ``load_processor()``.
In BLIP, you can also generate diverse captions by turning nucleus sampling on.

```python
from lavis.processors import load_processor
from lavis.models import load_model

# load image preprocesser used for BLIP
vis_processor = load_processor("blip_image_eval").build(image_size=384)
model = load_model(name="blip_caption", model_type="base_coco", is_eval=True, device=device)

image = vis_processor(image).unsqueeze(0).to(device)
model.generate({"image": raw_image}, use_nucleus_sampling=True)
# one generated random sample: ['some very pretty buildings and some water jets']
```


### Visual question answering (VQA)
*******************************
BLIP model is able to answer free-form questions about images in natural language.
To access the VQA model, simply replace the ``name`` and ``model_type`` arguments
passed to ``load_model_and_preprocess()``.

```python
from lavis.models import load_model_and_preprocess
model, vis_processors, txt_processors = load_model_and_preprocess(name="blip_vqa", model_type="vqav2", is_eval=True, device=device)

# ask a random question.
question = "Which city is this photo taken?"

image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
question = txt_processors["eval"](question)

model.predict_answers(samples={"image": image, "text_input": question}, inference_method="generate")
# ['singapore']
```

Unified Feature Extraction Interface

LAVIS provides a unified interface to extract multimodal features from each architecture.
To extract features, we load the feature extractor variants of each model.

```python
from lavis.models import load_model_and_preprocess

model, vis_processors, txt_processors = load_model_and_preprocess(name="blip_feature_extractor", model_type="base", is_eval=True, device=device)
caption = "a large fountain spewing water into the air"

image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
text_input = txt_processors["eval"](question)

sample = {"image": image, "text_input": [text_input]}

features_multimodal = model.extract_features(sample)
print(features_multimodal.multimodal_embeds.shape)
# torch.Size([1, 12, 768])

features_image = model(sample, mode="image")
print(features_image.image_embeds.shape)
# torch.Size([1, 197, 768])
print(features_image.image_features.shape)
# torch.Size([1, 197, 256])

features_text = model(sample, mode="text")
print(features_text.text_embeds.shape)
# torch.Size([1, 197, 768])
print(features_text.text_features.shape)
# torch.Size([1, 197, 256])
```

Since LAVIS supports a unified feature extraction interface, minimal changes are necessary to use a different model as feature extractor. For example,
to use ALBEF as the feature extractor, one only needs to change the following line:

```python
model, vis_processors, txt_processors = load_model_and_preprocess(name="albef_feature_extractor", model_type="base", is_eval=True, device=device)
```

Similarly, to use CLIP as feature extractor:

```python
model, vis_processors, txt_processors = load_model_and_preprocess(name="clip_feature_extractor", model_type="base", is_eval=True, device=device)
# model, vis_processors, txt_processors = load_model_and_preprocess(name="clip_feature_extractor", model_type="RN50", is_eval=True, device=device)
# model, vis_processors, txt_processors = load_model_and_preprocess(name="clip_feature_extractor", model_type="ViT-L-14", is_eval=True, device=device)
```

### Dataset Zoo
LAVIS inherently supports a wide variety of common language-vision datasets by providing automatic download scripts to help download and organize these datasets;
and implements PyTorch datasets for these datasets. To view supported datasets, use the following code:

```python
from lavis.datasets.builders import dataset_zoo
dataset_names = dataset_zoo.get_names()
print(dataset_names)
# ['aok_vqa', 'coco_caption', 'coco_retrieval', 'coco_vqa', 'conceptual_caption_12m',
#  'conceptual_caption_3m', 'didemo_retrieval', 'flickr30k', 'imagenet', 'laion2B_multi',
#  'msrvtt_caption', 'msrvtt_qa', 'msrvtt_retrieval', 'msvd_caption', 'msvd_qa', 'nlvr',
#  'nocaps', 'ok_vqa', 'sbu_caption', 'snli_ve', 'vatex_caption', 'vg_caption', 'vg_vqa']
print(len(dataset_names))
# 23
```
After downloading the images, we can use ``load_dataset()`` to obtain the dataset. On the first run, this will automatically download and cache annotation files.

```python
from lavis.datasets.builders import load_dataset
coco_dataset = load_dataset("coco_caption")

print(coco_dataset.key())
# dict_keys(['train', 'val', 'test'])

print(len(coco_dataset["train"]))
# 566747

print(coco_dataset["train"][0])
# {'image': <PIL.Image.Image image mode=RGB size=640x480>,
#  'text_input': 'A woman wearing a net on her head cutting a cake. ',
#  'image_id': 0}
```

If you already host a local copy of the dataset, you can pass in the ``vis_path`` argument to change the default location to load images.

```python
    coco_dataset = load_dataset("coco_caption", vis_path=YOUR_LOCAL_PATH)
```

## Installation

1. (Optional) Creating conda environment

```bash
conda create -n lavis python=3.8
conda activate lavis
```

2. Cloning and building from source

```bash
git clone https://github.com/MetaMind/LAVIS.git
cd LAVIS
pip install .
```

If you would like to develop on LAVIS, it is recommended to install in editable mode:
```bash
pip install -e .
```

## How to Contribute

We welcome the contribution from the open-source community to improve the library!

To add a new tasks, datasets and models into the library, please follow the template and steps demonstrated in this
[documentation]().

## Technical Report and Citing OmniXAI
You can find more details in our technical report: [TBD]()

If you're using LAVIS in your research or applications, please cite using this BibTeX:
```
@article{lavis,
  author    = {Dongxu Li and Junnan Li and Hung Le and Guangsen Wang and Silvio Savarese and Steven Hoi},
  title     = {LAVIS: A Library for Language-Vision Intelligence},
  year      = {2022},
}
```

## Contact Us
If you have any questions, comments or suggestions, please do not hesitate to contact us at [TBD]()

## License
[BSD 3-Clause License](LICENSE)
