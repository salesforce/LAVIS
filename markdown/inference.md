## Inference Examples
We first show examples of using LAVIS to run inference on customized data.

### Image Captioning
*******************************
In this example, we use the BLIP model to generate a caption for the image. To make inference even easier, we also associate each
pre-trained model with its preprocessors (transforms), accessed via ``load_model_and_preprocess()``.

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
