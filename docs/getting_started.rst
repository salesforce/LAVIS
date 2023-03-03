Dataset Zoo
##################
LAVIS inherently supports a wide variety of common language-vision datasets by providing automatic download scripts to help download and organize these datasets; 
and implements PyTorch datasets for these datasets. To view supported datasets, use the following code:

.. code-block:: python

    from lavis.datasets.builders import dataset_zoo
    dataset_names = dataset_zoo.get_names()
    print(dataset_names)
    # ['aok_vqa', 'coco_caption', 'coco_retrieval', 'coco_vqa', 'conceptual_caption_12m',
    #  'conceptual_caption_3m', 'didemo_retrieval', 'flickr30k', 'imagenet', 'laion2B_multi',
    #  'msrvtt_caption', 'msrvtt_qa', 'msrvtt_retrieval', 'msvd_caption', 'msvd_qa', 'nlvr',
    #  'nocaps', 'ok_vqa', 'sbu_caption', 'snli_ve', 'vatex_caption', 'vg_caption', 'vg_vqa']
    print(len(dataset_names))
    # 23


Auto-Downloading and Loading Datasets
######################################
We now take COCO caption dataset as an example to demonstrate how to download and prepare the dataset.

In ``lavis/datasets/download_scripts/``, we provide tools to download most common public language-vision datasets supported by LAVIS.
The COCO caption dataset uses images from COCO dataset. Therefore, we first download COCO images via:

.. code-block:: bash
    
    cd lavis/datasets/download_scripts/ && python download_coco.py

This will automatically download and extract COCO images to the default LAVIS cache location.
The default cache location is ``~/.cache/lavis``, defined in ``lavis/configs/default.yaml``.

After downloading the images, we can use ``load_dataset()`` to obtain the dataset. On the first run, this will automatically download and cache annotation files.

.. code-block:: python

    from lavis.datasets.builders import load_dataset
    coco_dataset = load_dataset("coco_caption")

    print(coco_dataset.keys())
    # dict_keys(['train', 'val', 'test'])

    print(len(coco_dataset["train"]))
    # 566747

    print(coco_dataset["train"][0])
    # {'image': <PIL.Image.Image image mode=RGB size=640x480>,
    #  'text_input': 'A woman wearing a net on her head cutting a cake. ',
    #  'image_id': 0}

If you already host a local copy of the dataset, you can pass in the ``vis_path`` argument to change the default location to load images.

.. code-block:: python

    coco_dataset = load_dataset("coco_caption", vis_path=YOUR_LOCAL_PATH)


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
    raw_image = Image.open("docs/_static/merlion.png").convert("RGB")

This example image shows `Merlion park <https://en.wikipedia.org/wiki/Merlion>`_ (`image credit <https://theculturetrip.com/asia/singapore/articles/what-exactly-is-singapores-merlion-anyway/>`_), a landmark in Singapore.

.. image:: _static/merlion.png

Image Captioning
*******************************
We now use the BLIP model to generate a caption for the image. To make inference even easier, we also associate each
pre-trained model with its preprocessors (transforms),  we use ``load_model_and_preprocess()`` with the following arguments:

- ``name``: The name of the model to load. This could be a pre-trained model, task model, or feature extractor. See ``model_zoo`` for a full list of model names.
- ``model_type``: Each architecture has variants trained on different datasets and at different scale. See Types column in ``model_zoo`` for a full list of model types.
- ``is_eval``: if `True`, set the model to evaluation mode. This is desired for inference or feature extraction.
- ``device``: device to load the model to.

.. code-block:: python

    from lavis.models import load_model_and_preprocess
    # loads BLIP caption base model, with finetuned checkpoints on MSCOCO captioning dataset.
    # this also loads the associated image processors
    model, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=device)

    # preprocess the image
    # vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

    # generate caption
    model.generate({"image": image})
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
To access the VQA model, simply replace the ``name`` and ``model_type`` arguments 
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


Unified Feature Extraction Interface
####################################

LAVIS provides a unified interface to extract multimodal features from each architecture.
To extract features, we load the feature extractor variants of each model.
The multimodal feature can be used for multimodal classification. The low-dimensional unimodal features can be used to compute cross-modal similarity.

.. code-block:: python

    from lavis.models import load_model_and_preprocess 
    
    model, vis_processors, txt_processors = load_model_and_preprocess(name="blip_feature_extractor", model_type="base", is_eval=True, device=device)
    caption = "a large fountain spewing water into the air"

    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    text_input = txt_processors["eval"](caption)

    sample = {"image": image, "text_input": [text_input]}

    features_multimodal = model.extract_features(sample)
    print(features_multimodal.keys())
    # odict_keys(['image_embeds', 'multimodal_embeds'])
    print(features_multimodal.multimodal_embeds.shape)
    # torch.Size([1, 12, 768]), use features_multimodal[:, 0, :] for multimodal classification tasks

    features_image = model.extract_features(sample, mode="image")
    print(features_image.keys())
    # odict_keys(['image_embeds', 'image_embeds_proj'])
    print(features_image.image_embeds.shape)
    # torch.Size([1, 197, 768])
    print(features_image.image_embeds_proj.shape)
    # torch.Size([1, 197, 256])

    features_text = model.extract_features(sample, mode="text")
    print(features_text.keys())
    # odict_keys(['text_embeds', 'text_embeds_proj'])
    print(features_text.text_embeds.shape)
    # torch.Size([1, 12, 768])
    print(features_text.text_embeds_proj.shape)
    # torch.Size([1, 12, 256])
    
    similarity = features_image.image_embeds_proj[:, 0, :] @ features_text.text_embeds_proj[:, 0, :].t()
    print(similarity)
    # tensor([[0.2622]])

Since LAVIS supports a unified feature extraction interface, minimal changes are necessary to use a different model as feature extractor. For example,
to use ALBEF as the feature extractor, one only needs to change the following line:

.. code-block:: python

    model, vis_processors, txt_processors = load_model_and_preprocess(name="albef_feature_extractor", model_type="base", is_eval=True, device=device)

Similarly, to use CLIP as feature extractor: 

.. code-block:: python

    model, vis_processors, txt_processors = load_model_and_preprocess(name="clip_feature_extractor", model_type="base", is_eval=True, device=device)
    # model, vis_processors, txt_processors = load_model_and_preprocess(name="clip_feature_extractor", model_type="RN50", is_eval=True, device=device)
    # model, vis_processors, txt_processors = load_model_and_preprocess(name="clip_feature_extractor", model_type="ViT-L-14", is_eval=True, device=device)
