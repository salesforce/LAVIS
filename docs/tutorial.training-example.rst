Example on Finetuning BLIP on COCO-Captioning
################################################

To finetune BLIP model on the coco caption dataset, first refer to :ref:`prep coco` to prepare the dataset if you have not done so.

To finetune the model, we have prepared a run script for you, which can run as follows:

.. code-block:: bash

    bash run_scripts/blip/train/train_caption_coco_large.sh

This will finetune the pre-trained BLIP large model into a new model that can be used for captioning.

Deep Dive
**********
Now let's take a closer look at the script and see what it does.

.. code-block:: bash

    python -m torch.distributed.run --nproc_per_node=8 train.py --cfg-path lavis/projects/blip/train/caption_coco_large_ft.yaml

As can be seen, the script simply calls the :code:`train.py` with PyTorch distributed training enabled.
The :code:`--cfg-path` argument specifies the **runtime config** file to use. The config file is a YAML file that specifies the training parameters, shown as follows:

.. literalinclude:: ../lavis/projects/blip/train/caption_coco_large_ft.yaml
    :language: yaml
    :linenos:

The runtime config file is divided into 3 sections:
    - :code:`model`: specifies the model architecture and type to use.
    - :code:`data`: specifies the dataset to use.
    - :code:`run`: specifies the runner arguments, such as tasks, optimizer, learning rate scheduler, etc.

We describe each section in detail below.

Model configurations
=====================

.. literalinclude:: ../lavis/projects/blip/train/caption_coco_large_ft.yaml
    :language: yaml
    :linenos:
    :lines: 6-10

The :code:`arch` argument specifies the model architecture to use. In this case, we use the :code:`blip_caption` architecture.
You can find available architectures by inspecting the :code:`model_zoo`.
Once the architecture is specified, the runner will look for the model class registered with the name and try to instantiate a model instance.
In this case :code:`BlipCaption` is the model registered with the name :code:`blip_caption`.

The registry maintains a mapping from the name string to the model class.
This allows the runner to find the model class dynamically based on the name string from the config file. 
The following segment in :code:`lavis/models/blip_models/blip_caption.py` shows how :code:`BlipCaption` is registered with the name string :code:`blip_caption`:

.. literalinclude:: ../lavis/models/blip_models/blip_caption.py
    :language: python
    :linenos:
    :lines: 20-38

One same model architecture may be pre-trained or finetuned on different datasets or have different model configurations.
For example, :code:`BlipCaption` have:

    - :code:`base_coco`: pre-trained base BLIP model adapated for COCO captioning finetuning.

    - :code:`large_coco`: pre-trained large BLIP model adapated for COCO captioning finetuning.

Therefore, we also need to specify :code:`model_type`. Here we use :code:`large_coco`.
And we set :code:`load_finetuned` to :code:`False` to indicate that we are finetuning the model from the pre-trained weights.
If :code:`load_finetuned` set to :code:`True` as by default, the model will load finetuned weights on coco captioning.

Given the model architecture and type, the library will then look for the default model config for :code:`large_coco` in :code:`lavis/models/blip_models/blip_caption.py`.
As can be seen in the above code snippet, the corresponding config path is stored in :code:`BlipCaption.PRETRAINED_MODEL_CONFIG_DICT`. 
Then the library will load :code:`lavis/configs/models/blip_caption_large_coco.yaml` as the configuration to build the model.

*Priority of Configs*: Note that the priority of the run config is higher than the default model config, meaning that arguments in the run config will override the default model config.
For example, in the default model config, :code:`load_finetuned` is set to :code:`True` by default, while in the run config, we set it to :code:`False` and finetuning from the pre-trained weights only.


Dataset configurations
=========================

The second section of the config file specifies the dataset(s) to use.

.. literalinclude:: ../lavis/projects/blip/train/caption_coco_large_ft.yaml
    :language: yaml
    :linenos:
    :lines: 12-24

We associate each dataset with a :code:`vis_processor` and a :code:`text_processor`, responsible for processing the visual and textual input respectively.
Here we again use the registry mechanism to dynamically load the processor class based on the name string.
For example, :code:`blip_image_train` is the name string for the :code:`BlipImageTrainProcessor` class, which is registered in :code:`lavis/processors/blip_processors.py`.

Similarly, the dataset name string is also registered in the registry, pointing to a dataset builder :code:`COCOCapBuilder` class.
By default, the builder will load the default dataset configuration as in :code:`DATASET_CONFIG_DICT`. You may also add new dataset types by adding new entries to the dictionary.

The dataset configuration used here is:

.. literalinclude:: ../lavis/configs/datasets/coco/defaults_cap.yaml
    :language: yaml
    :linenos:
    :lines: 6-28

In this configuration file, we specify the dataset name and mainly its building information.
The build information is divided into two parts: :code:`annotation` and :code:`images`. The annotation files will be automatically downloaded upon loading the dataset for the first time.
The :code:`images` part specifies the image root directory. This is a relative path to the cache directory, which is :code:`cache` by default. If you have a local copy of the dataset, you can specify the path to the local copy by
overwriting the :code:`images` part in the runtime config file. For example, you may alter the run config as below to use your local dataset copy:

.. code:: yaml

    datasets:
        coco_caption: # name of the dataset builder
            vis_processor:
                train:
                name: "blip_image_train"
                eval:
                name: "blip_image_eval"
            text_processor:
                train:
                name: "blip_caption"
                prompt: "a picture of "
                eval:
                name: "blip_caption"
            images:
                YOUR_LOCAL_IMAGE_ROOT_DIR

LAVIS supports using multiple datasets for training. See an example in :code:`lavis/projects/blip/train/pretrain_14m.yaml`.


Runner configurations
=========================
The last section of the config file specifies the arguments for the runner, shown below:

.. literalinclude:: ../lavis/projects/blip/train/caption_coco_large_ft.yaml
    :language: yaml
    :linenos:
    :lines: 26-56

Here we specify runner-related arguments, including
    - task-specific arguments, such as :code:`task`, :code:`max_len`, :code:`min_len`, etc.
    - learning rate schedulers, optimizer;
    - distributed training settings;
    - logging and checkpointing settings.

Available Configurations
#########################

See :ref:`config` for the full list of available configurations and their descriptions.
