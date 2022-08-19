Model Zoo
####################################
LAVIS supports a growing list of pre-trained models for different tasks,
datatsets and of varying sizes. Let's get started by viewing the supported models.

.. code-block:: python

    from lavis.models import model_zoo
    print(model_zoo)
    # ==================================================
    # Architectures                  Types
    # ==================================================
    # albef_classification           base, ve
    # albef_nlvr                     base
    # albef_pretrain                 base
    # albef_retrieval                base, coco, flickr
    # albef_vqa                      base, vqav2
    # alpro_qa                       base, msrvtt, msvd
    # alpro_retrieval                base, msrvtt, didemo
    # blip_caption                   base, base_coco, large, large_coco
    # blip_classification            base
    # blip_feature_extractor         base
    # blip_nlvr                      base
    # blip_pretrain                  base
    # blip_retrieval                 base, coco, flickr
    # blip_vqa                       base, vqav2
    # clip                           ViT-B-32, ViT-B-16, ViT-L-14, ViT-L-14-336, RN50

    # show total number of support model variants
    len(model_zoo)
    # 33


Inference with Pre-trained Models
####################################

Now let's see how to use models in LAVIS to perform inference on example data. We first
load a sample image from local.

.. code-block:: python

    from PIL import Image

    # setup device to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load sample image
    raw_image = Image.open("docs/data/merlion.png").convert("RGB")

This example image shows `Merlion park <https://en.wikipedia.org/wiki/Merlion>`_ (`image credit <https://theculturetrip.com/asia/singapore/articles/what-exactly-is-singapores-merlion-anyway/>`_), a landmark in Singapore.

.. image:: ../data/merlion.png

Image Captioning
*******************************
We now use the BLIP model to generate a caption for the image. To make inference even easier, we also associate each
pre-trained model with its preprocessors (transforms),  we use ``load_model_and_preprocess()`` with the following parameters:

- ``name``: The name of the model to load. This could be a pre-trained model, task model, or feature extractor. See ``model_zoo`` for a full list of model names.
- ``model_type``: Each architecture has variants trained on different datasets and at different scale. See Types column in ``model_zoo`` for a full list of model types.
- ``is_eval``: if `True`, set the model to evaluation mode. This is desired for inference or feature extraction.
- ``devce``: device to load the model to.

.. code-block:: python

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


You may also load models and their preprocessors separately via ``load_model()`` and ``load_processor()``.
In BLIP, you can also generate diverse captions by turning nucleus sampling on.

.. code-block:: python

    from lavis.processors import load_processor
    from lavis.models import load_model

    # load image preprocesser used for BLIP
    vis_processor = load_processor("blip_image_eval").build(image_size=384)
    model = load_model(name="blip_caption", model_type="base_coco", is_eval=True, device=device)

    image = vis_processor(image).unsqueeze(0).to(device)
    model.generate({"image": raw_image}, use_nucleus_sampling=True)
    # one generated random sample: ['some very pretty buildings and some water jets']


Visual question answering (VQA)
*******************************
BLIP model is able to answer free-form questions about images in natural language.
To access the VQA model, simply replace the ``name`` and ``model_type`` parameters 
passed to ``load_model_and_preprocess()``.

.. code-block:: python

    from lavis.models import load_model_and_preprocess
    model, vis_processors, txt_processors = load_model_and_preprocess(name="blip_vqa", model_type="vqav2", is_eval=True, device=device)

    # ask a random question.
    question = "Which city is this photo taken?"
    
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    question = txt_processors["eval"](question)

    model.predict_answers(samples={"image": image, "text_input": question}, inference_method="generate")
    # ['singapore']


Feature Extraction
*******************************

LAVIS provides a unified interface to extract multimodal features from each architecture.
To extract features, we load the feature extractor variants of each model.

.. code-block:: python

    from lavis.models import load_model_and_preprocess 
    
    model, vis_processors, txt_processors = load_model_and_preprocess(name="blip_feature_extractor", model_type="base", is_eval=True, device=device)
    caption = "a large fountain spewing water into the air"

    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    text_input = txt_processors["eval"](question)

    sample = {"image": image, "text_input": [text_input]}

    features_multimodal = model.extract_features(sample)
    print(features_multimodal.keys())
    # odict_keys(['image_embeds', 'multimodal_embeds'])
    print(features_multimodal.multimodal_embeds.shape)
    # torch.Size([1, 12, 768])

    features_image = model(sample, mode="image")
    print(features_image.keys())
    # odict_keys(['image_embeds', 'image_features'])
    print(features_image.image_embeds.shape)
    # torch.Size([1, 197, 768])
    print(features_image.image_features.shape)
    # torch.Size([1, 197, 256])

    features_text = model(sample, mode="text")
    print(features_text.keys())
    # odict_keys(['text_embeds', 'text_features'])
    print(features_text.text_embeds.shape)
    # torch.Size([1, 197, 768])
    print(features_text.text_features.shape)
    # torch.Size([1, 197, 256])

Since LAVIS supports a unified feature extraction interface, minimal changes are necessary to use a different model as feature extractor. For example,
to use ALBEF as the feature extractor, one only needs to change the following line:

.. code-block:: python

    model, vis_processors, txt_processors = load_model_and_preprocess(name="albef_feature_extractor", model_type="base", is_eval=True, device=device)

Similarly, to use CLIP as feature extractor: 

.. code-block:: python

    model, vis_processors, txt_processors = load_model_and_preprocess(name="clip_feature_extractor", model_type="base", is_eval=True, device=device)
    # model, vis_processors, txt_processors = load_model_and_preprocess(name="clip_feature_extractor", model_type="RN50", is_eval=True, device=device)
    # model, vis_processors, txt_processors = load_model_and_preprocess(name="clip_feature_extractor", model_type="ViT-L-14", is_eval=True, device=device)

Evaluating Pre-trained Models on Task Datasets
###############################################
LAVIS provides pre-trained and finetuned model for off-the-shelf evaluation on task dataset. 
Let's now see an example to evaluate BLIP model on the captioning task, using MSCOCO dataset.

Preparing Datasets
******************
First, let's download the dataset. LAVIS provides `automatic downloading scripts` to help prepare 
most of the public dataset, to download MSCOCO dataset, simply run

.. code-block:: bash

    cd lavis/datasets/download_scripts && bash download_coco.py

This will put the downloaded dataset at a default cache location ``~/.cache/lavis`` used by LAVIS.

Evaluating pre-trained models
******************************

To evaluate pre-trained model, simply run

.. code-block:: bash

    bash run_scripts/lavis/blip/eval/eval_coco_cap.sh

Or to evaluate a large model:

.. code-block:: bash

    bash run_scripts/lavis/blip/eval/eval_coco_cap_large.sh

Fine-tuning Pre-trained Models on Task Datasets
###############################################
LAVIS provides scripts to pre-train and finetune supported models on standard language-vision tasks, stored at ``lavis/run_scripts/``. 
To replicate the experiments, just run these bash scripts. For example, to train BLIP model on the image-text retrieval task with MSCOCO dataset, we can run

.. code-block::

    bash run_scripts/lavis/blip/train/train_retrieval_coco.sh

Inside the scripts, we can see 

.. code-block:: bash

    python -m torch.distributed.run --nproc_per_node=8 train.py --cfg-path lavis/projects/blip/train/retrieval_coco_ft.yaml

where we start a pytorch distributed training on 8 GPUs (you may change according to your own hardware setup). The ``--cfg-path`` specifys a `runtime configuration file`, specifying
the task, model, dataset and training recipes. 
