## BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models
This is the official implementation of the <a href="https://arxiv.org/abs/2201.12086">BLIP-2 paper</a>. We integrate BLIP-2 into LAVIS. 

### Install:
```
pip install salesforce-lavis
```
or install from source following LAVIS instruction.

### Demo:
Try out our [Colab demo](https://colab.research.google.com/github/salesforce/LAVIS/blob/main/examples/blip2_instructed_generation.ipynb)!
Or run the [notebook](https://github.com/salesforce/LAVIS/blob/main/examples/blip2_instructed_generation.ipynb) on your own machine.


### BLIP-2 Model Zoo 
```python
# ==================================================
# Architectures                  Types
# ==================================================
# blip2_opt                      pretrain_opt2.7b, caption_coco_opt2.7b, pretrain_opt6.7b, caption_coco_opt6.7b
# blip2_t5                       pretrain_flant5xl, caption_coco_flant5xl, pretrain_flant5xxl
# blip2                          pretrain, coco
```

### Inference Example
Let’s see how to use BLIP-2 models to perform zero-shot instructed image-to-text generation. We first load a sample image from local.
```python
import torch
from PIL import Image
# setup device to use
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
# load sample image
raw_image = Image.open("../docs/_static/merlion.png").convert("RGB")
display(raw_image.resize((596, 437)))
```

Then we load a pre-trained BLIP-2 model with its preprocessors (transforms).
```python
import torch
from lavis.models import load_model_and_preprocess
# loads BLIP-2 pre-trained model
model, vis_processors, _ = load_model_and_preprocess(name="blip_t5", model_type="pretrain_flant5xxl", is_eval=True, device=device)
# prepare the image
image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
```

Given the image and a text prompt, ask the model to generate the response.
```python
model.generate({"image": image, "prompt": "Question: which city is this? Answer:"})
# 'singapore'
```

Ask the model to explain its answer.
```python
model.generate({
    "image": image,
    "prompt": "Question: which city is this? Answer: singapore. Question: why?"})
# 'it has a statue of a merlion'    
```


Ask a follow-up question.
```python
# prepare context prompt
context = [
    ("which city is this?", "singapore"),
    ("why?", "it has a statue of a merlion"),
]
question = "where is the name merlion coming from?"
template = "Question: {} Answer: {}."
prompt = " ".join([template.format(context[i][0], context[i][1]) for i in range(len(context))]) + " Question: " + question + " Answer:"
print(prompt)
# generate model's response
model.generate({"image": image,"prompt": prompt})
# 'merlion is a portmanteau of mermaid and lion'
```

### Benchmark Evaluation 
Follow [Dataset Download](https://opensource.salesforce.com/LAVIS//latest/getting_started.html#auto-downloading-and-loading-datasets) to prepare common vision-language datasets.

Run [these scripts](https://github.com/salesforce/LAVIS/tree/main/run_scripts/blip2/eval) for evaluating pretrained and finetuned models. 

For model training, please follow LAVIS documentation.
