Adding Datasets
################################################

This is a tutorial on adding a new dataset using ``lavis.datasets`` module. 

The LAVIS library includes a standard dataset module, which allows customization to add new datasets. 
The ``lavis.datasets`` module is designed such that any new dataset class can be easily added and adapted from our code base, including creating dataset configuration, and defining and associating new dataset classes.

In this tutorial, we will replicate the steps to add a dataset class for the `Audio-Visual Scene-Aware Dialogue (AVSD) <https://arxiv.org/pdf/1901.09107.pdf>`_ benchmark for the video-grounded dialogue task.

Dataset Configuration ``lavis.configs.datasets``
**************************************************************

First, we define the basic configurations for this dataset, including a new dataset class ``avsd_dialogue``, dataset card, and data types. 
We can define any new dataset configuration in ``lavis.configs.datasets``. For instance, under this module, we can set up a configuration file ``avsd/defaults_dial.yaml`` as follows:  

.. code-block:: yaml

    datasets:
      avsd_dialogue: # name of the dataset builder
        dataset_card: dataset_card/avsd_dialogue.md # path to the dataset card 
        data_type: features # [images|videos|features] we use features in this case for extracted video features 

        build_info:
          # Be careful not to append minus sign (-) before split to avoid itemizing
          annotations:
            train:
              url: /export/home/data/avsd/train_set4DSTC7-AVSD.json
              storage: avsd/annotations/train.json
            val:
              url: /export/home/data/avsd/valid_set4DSTC7-AVSD.json
              storage: avsd/annotations/val.json 
            test:
              url: /export/home/data/avsd/test_set4DSTC7-AVSD.json
              storage: avsd/annotations/test.json 
          features:
            storage: /export/home/data/avsd/features/ 


Dataset Card
===============
One optional step to set up dataset configuration is defining a dataset card, which contains more details about the dataset such as description, tasks, and metrics. 
For instance, we can define a dataset card for the AVSD benchmark in ``dataset_card/avsd_dialogue.md``.
Depending on the dataset, we included in its corresponding dataset card the command for auto-downloading data (with python code defined in ``lavis.datasets.download_scripts``) that will automatically load the data and store it in a specific folder.
Else, you should describe in the dataset card the external download instructions from the original data source to load the dataset properly. 

One example of a dataset card for the AVSD benchmark is: 

.. code-block:: md

    ![Samples from the AVSD dataset (Image credit: "https://arxiv.org/pdf/1901.09107.pdf").](imgs/avsd_dialogue.png)(Samples from the AVSD dataset. Image credit: "https://arxiv.org/pdf/1901.09107.pdf")
    
    # Audio-Visual Scene-Aware Dialogues (AVSD) 
    
    ## Description
    [Audio-Visual Scene-Aware Dialogues (AVSD)](https://github.com/hudaAlamri/DSTC7-Audio-Visual-Scene-Aware-Dialog-AVSD-Challenge) contains more than 10,000 dialogues, each of which is grounded on a unique video. In the test split, for each test sample, 6 reference dialogue responses are provided. 
    
    
    ## Task
    
    (https://github.com/hudaAlamri/DSTC7-Audio-Visual-Scene-Aware-Dialog-AVSD-Challenge)
    
    In a **video-grounded dialogue task**, the system must generate responses to user input in the context of a given dialog.
    This context consists of a dialog history (previous utterances by both user and system) in addition to video and audio information that comprise the scene. The quality of a system’s automatically generated sentences is evaluated using objective measures to determine whether or not the generated responses are natural and informative
    
    ## Metrics
    Models are typically evaluated according to [BLEU](https://aclanthology.org/P02-1040/), [CIDER](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Vedantam_CIDEr_Consensus-Based_Image_2015_CVPR_paper.pdf), [METEOR](https://aclanthology.org/W05-0909/), and [ROUGE-L](https://aclanthology.org/W04-1013/) metrics. 
    
    ## Leaderboard
    
    ....
    
    
    ## Auto-Downloading
    
    Please refer to [benchmark webite](https://github.com/hudaAlamri/DSTC7-Audio-Visual-Scene-Aware-Dialog-AVSD-Challenge) for instructions to download the dataset. 
    
    
    ## References
    "Audio Visual Scene-Aware Dialog", Huda Alamri, Vincent Cartillier, Abhishek Das, Jue Wang, Anoop Cherian, Irfan Essa, Dhruv Batra, Tim K. Marks, Chiori Hori, Peter Anderson, Stefan Lee, Devi Parikh

Visual Data Type
==============================
We currently limit the visual data types to one of three options: ``images``, ``videos``, and ``features``. 
"Images" and "videos" refer to the raw visual data, which is appropriate for models processing visual data in their original forms (e.g. ViT models). 
"Features" are visual representations extracted from pretrained models (e.g. CNN models). 
In this tutorial, the AVSD benchmark consists of video features extracted from 3D-CNN models. 

Build Info
==============================
Build info refers to the specific locations where data is stored and cached. 

For text annotations (e.g. captioning or dialogues), by default, we include three data splits, namely "train", "val", and "test", typically used in all machine learning projects. 
For each split, we specify 2 parameters: ``url``  and ``storage``.
``url`` can be either an online URL where the dataset can be loaded automatically (e.g. from *googleapis*), or a local directory where data is already downloaded beforehand. 
``storage`` is the directory where the data will be cached over time, avoiding downloading data repeatedly.

For visual data annotations, ensure the field name matches the data types defined earlier (e.g. one of "images", "videos" or features"). 
As visual features are usually large and should be downloaded beforehand, we maintain only a ``storage`` parameter where visual data is cached. 

Dataset ``lavis.datasets.datasets``
**************************************************************

Base Dataset ``lavis.datasets.datasets.base_dataset``
=======================================================
In this step, we want to define new dataset classes that inherit our base dataset class ``lavis.datasets.datasets.base_dataset``. This base dataset class already defines standard methods such as ``collater`` which uses the default collator from Pytorch. 

.. code-block:: python

    import json
    from typing import Iterable
    
    from torch.utils.data import Dataset, ConcatDataset
    from torch.utils.data.dataloader import default_collate
        
    class BaseDataset(Dataset):
        def __init__(
            self, vis_processor=None, text_processor=None, vis_root=None, ann_paths=[]
        ):
            """
            vis_root (string): Root directory of images (e.g. coco/images/)
            ann_root (string): directory to store the annotation file
            """
            self.vis_root = vis_root
    
            self.annotation = []
            for ann_path in ann_paths:
                self.annotation.extend(json.load(open(ann_path, "r")))
    
            self.vis_processor = vis_processor
            self.text_processor = text_processor
    
            self._add_instance_ids()
    
        def __len__(self):
            return len(self.annotation)
    
        def collater(self, samples):
            return default_collate(samples)
    
        def set_processors(self, vis_processor, text_processor):
            self.vis_processor = vis_processor
            self.text_processor = text_processor
    
        def _add_instance_ids(self, key="instance_id"):
            for idx, ann in enumerate(self.annotation):
                ann[key] = str(idx)

Any dataset subclass will inherit these methods and it is optional to define and overwrite these methods accordingly to the specifications of the dataset. 
We encourage users not to modify the base dataset class as any modification will have cascading impacts on any other dataset classes that inherit this base dataset. 
Instead, the users should independently create new dataset classes to cater to their specific requirements. 

Dialogue Datasets ``lavis.datasets.datasets.dialogue_datasets``
======================================================================

For example, for the AVSD dataset, we want to define a new dataset subclass ``DialogueDataset`` for dialogue tasks. We can define this dataset class in ``lavis.datasets.datasets.dialogue_datasets`` as following: 

.. code-block:: python

    import os
    from collections import OrderedDict
        
    from lavis.datasets.datasets.base_dataset import BaseDataset
    
    import json 
    import copy 

    class DialogueDataset(BaseDataset):
        def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
            """
            vis_processor (string): visual processor 
            text_processor (string): textual processor 
            vis_root (string): Root directory of images (e.g. coco/images/)
            ann_paths (string): Root directory of images (e.g. coco/images/)
            """
                
            self.vis_root = vis_root
    
            self.annotation = []
            for ann_path in ann_paths:
                dialogs = json.load(open(ann_path, "r"))['dialogs']
                for dialog in dialogs: 
                    all_turns = dialog['dialog']
                    dialogue_context = [] 
                    for turn in all_turns: 
                        dialog_instance = copy.deepcopy(dialog)
                        question = turn['question']
                        answer = turn['answer'] 
                        
                        dialog_instance['dialog'] = copy.deepcopy(dialogue_context) 
                        dialog_instance['question'] = question
                        dialog_instance['answer'] = answer 
                        self.annotation.append(dialog_instance)
                        dialogue_context.append(turn)
                        
            self.vis_processor = vis_processor
            self.text_processor = text_processor
    
            self._add_instance_ids()
    
            self.img_ids = {}
            n = 0
            for ann in self.annotation:
                img_id = ann["image_id"]
                if img_id not in self.img_ids.keys():
                    self.img_ids[img_id] = n
                    n += 1

Class inheritance allows us to define multiple subclasses. For instance, we want another dialogue dataset class that is defined only for the test split. We can define another dataset class ``DialogueEvalDataset`` as similarly defined above but the annotations are processed differently. 
Typically, in dialogue tasks, during test time, only a single test sample is constructed per dialogue (rather than decomposing all dialogue turns as samples during training time).
The dataset class can then be defined as: 

.. code-block:: python

    class DialogueEvalDataset(BaseDataset):
        def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
            # ...
            # defined similarly as DialogueDataset above 
            # except for the loading of dialogue annotation data            
    
            self.annotation = []
            for ann_path in ann_paths:
                dialogs = json.load(open(ann_path, "r"))['dialogs']
                for dialog in dialogs: 
                    all_turns = dialog['dialog']
                    dialogue_context = all_turns[:-1]
                    last_turn = all_turns[-1] 
                    
                    question = last_turn['question']
                    answer = last_turn['answer'] 
                        
                    dialog['dialog'] = dialogue_context
                    dialog['question'] = question
                    dialog['answer'] = answer
                                        
                    self.annotation.append(dialog)


Using class inheritance to define datasets also allows us to develop more fine-grain class implementations, each of which is specifically designated for a benchmark. 
For instance, under the dialogue-based tasks, we can further define another dataset subclass that is specified for the AVSD dataset. 
We can define a new class ``AVSDDialDataset`` that further specifies how to load individual samples and collate them accordingly to specific requirements: 

.. code-block:: python

    import os
    from lavis.datasets.datasets.base_dataset import BaseDataset
    from lavis.datasets.datasets.dialogue_datasets import DialogueDataset, DialogueEvalDataset
    
    import torch 
        
    class AVSDDialDataset(DialogueDataset):
        def __init__(self, vis_processor, text_processor, vis_root, ann_paths):

            super().__init__(vis_processor, text_processor, vis_root, ann_paths)
    
        def __getitem__(self, index):
    
            ann = self.annotation[index]
    
            vname = ann["image_id"]
    
            video = self.vis_processor(self.vis_root, vname)
            
            dialogue = self.text_processor(ann)
            
            return {
                "video_fts": video['video_fts'],
                "video_token_type_ids": video['token_type_ids'], 
                "input_ids": dialogue['input_ids'], 
                "token_type_ids": dialogue['token_type_ids'],
                "labels": dialogue['labels'], 
                "image_id": ann["image_id"],
                "instance_id": ann["instance_id"]
            }
        
        def collater(self, samples):
            
            input_ids, token_type_ids, labels, video_fts, video_token_type_ids = [], [], [], [], []
            
            for i in samples:
                input_ids.append(i['input_ids'])
                token_type_ids.append(i['token_type_ids'])
                labels.append(i['labels'])
                video_fts.append(i['video_fts'])
                video_token_type_ids.append(i['video_token_type_ids'])
    
            input_ids = self.text_processor.padding(input_ids)
            
            labels = self.text_processor.padding(labels, -1)
            video_fts = self.vis_processor.padding(video_fts)
            
            token_type_ids = self.text_processor.padding(token_type_ids)
            video_token_type_ids = self.text_processor.padding(video_token_type_ids)
            token_type_ids = torch.cat([video_token_type_ids, token_type_ids], dim=1)
            
            attn_mask = self.text_processor.get_attention_mask(input_ids)
            video_mask = self.vis_processor.get_attention_mask(video_fts)
            attn_mask = torch.cat([video_mask, attn_mask], dim=1)
            
            video_labels = torch.ones((video_fts.size(0), video_fts.size(1))).long() * -1 # ignore token indice -1 by default 

            labels = torch.cat([video_labels, labels], dim=1)
            
            samples = {}
            samples['input_ids'] = input_ids
            samples['token_type_ids'] = token_type_ids
            samples['labels'] = labels
            samples['video_fts'] = video_fts
            samples['attn_mask'] = attn_mask
            
            return samples  

Note that in a dataset subclass, if methods such as ``__getitem__`` and ``collater`` are not defined, the same functions from the corresponding superclass will be used. 
For instance, by default, we always use the collater from the ``BaseDataset`` class to collate data samples. 

Dataset Builder ``lavis.datasets.builders``
**************************************************************
Dataset Builder is the data processing module that controls the dataset classes (by training or evaluation split) and associates the specific dataset configurations to these dataset classes. 

Base Dataset Builder ``lavis.datasets.builders.base_dataset_builder``
======================================================================

Note that any new builder class definition should inherit the base dataset builder class ``lavis.datasets.builders.base_dataset_builder``:

.. code-block:: python

    class BaseDatasetBuilder:
        train_dataset_cls, eval_dataset_cls = None, None
        ...

This allows us to standardize the operations of dataset builders across all builder classes. We advise the users to carefully review the standard methods defined in the base builder class, including methods such as ``_download_data`` and ``build_dataset`` that will load download the data and create instances of dataset classes: 

.. code-block:: python

    class BaseDatasetBuilder:
    ...

        def build_datasets(self):
            # download, split, etc...
            # only called on 1 GPU/TPU in distributed
    
            if is_main_process():
                self._download_data()
    
            if is_dist_avail_and_initialized():
                dist.barrier()
    
            # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
            logging.info("Building datasets...")
            datasets = self.build()  # dataset['train'/'val'/'test']
            
            return datasets
    
        def _download_data(self):
            self._download_ann()
            self._download_vis()
    
We encourage users not to modify the implementation of the base dataset builder class as this will affect all existing dataset builder subclasses.

Dialogue Dataset Builder ``lavis.datasets.builders.dialogue_builder``
======================================================================
We can define any new builder subclass and associate this builder with the corresponding dataset classes and dataset configurations. 
For instance, for the AVSD dataset, we can define a builder ``lavis.datasets.builders.dialogue_builder`` for dialogue-based datasets as follows: 

.. code-block:: python

    from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
    from lavis.datasets.datasets.avsd_dialogue_datasets import (
        AVSDDialDataset, 
        AVSDDialEvalDataset 
    )
    
    from lavis.common.registry import registry
    
    
    @registry.register_builder("avsd_dialogue")
    class AVSDDialBuilder(BaseDatasetBuilder):
        train_dataset_cls = AVSDDialDataset 
        eval_dataset_cls = AVSDDialEvalDataset 
    
        DATASET_CONFIG_DICT = {
            "default": "configs/datasets/avsd/defaults_dial.yaml"
        }

Note that we chose to separately define the parameters ``train_dataset_cls`` and  ``eval_dataset_cls`` to consider cases where data is processed differently between training and test time. 
For instance, in captioning tasks, during test time, each data sample often includes multiple ground-truth captions rather than just a single ground-truth during training time. 
If the data processing is the same in both training and test time, the two parameters can be linked to the same dataset class. 

Finally, define ``DATASET_CONFIG_DICT`` to associate the dataset configurations to the assigned dataset classes. 

Registering Builder ``lavis.datasets.builders.__init__``
======================================================================

To add a new builder class, ensure to first include the class within the ``__init__.py``. For instance, to define a new builder for the AVSD dataset: 

.. code-block:: python

    from lavis.datasets.builders.dialogue_builder import (
        AVSDDialBuilder
    )
    
    __all__ = [
        ...,
        "AVSDDialBuilder"
    ]

Assigning Builder 
======================================================================
Note that during data loading and processing, the builder being assigned must have the correct registry to be able to load it properly. 
For instance, the following should be specified in a configuration file e.g. ``dialogue_avsd_ft.yaml``: 

.. code-block:: yaml

    datasets:
      avsd_dialogue: # name of the dataset builder
        ...
        # processor configuration 
        ...

Subsequently, any processes (e.g. training) should load this configuration file to assign the correct builder which will then associate the correct dataset classes to construct data samples. 

.. code-block:: sh

    python train.py --cfg-path dialogue_avsd_ft.yaml
