.. toctree::
   :maxdepth: 3
   :caption: Tutorials 


Adding New Datasets ``lavis.datasets``
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

Adding New Processors ``lavis.processors``
################################################

This is a tutorial on adding new processors using ``lavis.processors`` module. 

The LAVIS library includes a standard processor module that preprocesses data e.g. image transformation and sequence concatenation.
The ``lavis.processors`` module is designed such that any processors can be added, specifically to the requirements of corresponding models of interest. 
In this tutorial, we will replicate the steps to add visual and textual processors specifically for `video-grounded dialogue tasks <https://arxiv.org/pdf/1901.09107.pdf>`_. 
In addition, we also want the processors to have processing features to make the data samples compatible with GPT-style models.

Base Processor ``lavis.processors.base_processors``
*****************************************************

Note that any new processor definition should inherit the base processor class ``BaseProcessor``:

.. code-block:: python

    from omegaconf import OmegaConf
    
    class BaseProcessor:
        def __init__(self):
            self.transform = lambda x: x
            return
    
        def __call__(self, item):
            return self.transform(item)
    
        @classmethod
        def from_config(cls, cfg=None):
            return cls()
    
        def build(self, **kwargs):
            cfg = OmegaConf.create(kwargs)
    
            return self.from_config(cfg)

This allows us to standardize operations of processors across all processor classes while still allowing customization of processors specifically to data and model types. 
We encourage users not to modify the implementation of the base processor class as this will have an impact on all existing processor subclasses.

GPT-style Processors ``lavis.processors.gpt_processors``
**************************************************************
In this step, we can define new processor classes, e.g. under ``lavis.processors.gpt_processors``, for GPT models designed specifically for video-grounded dialogues. 
First, we want to process video features by defining ``GPTVideoFeatureProcessor`` class.
In this tutorial, we assume video features are extracted beforehand and this processor simply loads the features from ``npy`` files.
Other methods that are specifically defined are ``padding`` (which is used by dataset instances to pad multiple video samples) and ``get_attention_mask`` (which creates an attention mask for Transformer attention in GPT models). 

.. code-block:: python 

    SPECIAL_TOKENS_DICT = {'bos_token': "<bos>", 'eos_token': "<eos>", 'additional_special_tokens': ["<speaker1>", "<speaker2>", "<video>", "<cap>"], 'pad_token': "<pad>"}
    ...

    @registry.register_processor("gpt_video_ft")
    class GPTVideoFeatureProcessor(BaseProcessor):
        def __init__(self, visual_ft, audio_ft):

            self.visual_ft = visual_ft
            self.audio_ft = audio_ft

            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self.tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT) 
                    
        def padding(self, seq):
            padded_seq = torch.nn.utils.rnn.pad_sequence(seq, batch_first=True, padding_value=1.0) 
            return padded_seq
        
        def get_attention_mask(self, seq):
            return torch.sum(seq != 1, dim=2) != 0
    
        def __call__(self, ft_root, vname):
            all_ft = []
            
            for ft_name in self.visual_ft:
                ft_path = os.path.join(ft_root, ft_name, vname)
                all_ft.append(np.load(ft_path + '.npy'))
            
            for ft_name in self.audio_ft: 
                ft_path = os.path.join(ft_root, ft_name, vname)
                all_ft.append(np.load(ft_path + '.npy'))
            
            min_len = min([len(ft) for ft in all_ft])
            
            sampled_ft = [ft[:min_len] for ft in all_ft]
            sampled_ft = np.concatenate(sampled_ft, axis=1)
            item = {} 
            item['video_fts'] = torch.Tensor(sampled_ft) 
            
            video_type_token = self.tokenizer.convert_tokens_to_ids('<video>')
            item['token_type_ids'] = torch.Tensor([video_type_token] * len(sampled_ft)).long() 
            
            return item 
    
        @classmethod
        def from_config(cls, cfg=None):
            if cfg is None:
                cfg = OmegaConf.create()
            
            visual_ft = cfg.get("visual_ft", ["i3d_rgb"])
            audio_ft = cfg.get("audio_ft", ["vggish"])
            
            return cls(
                visual_ft=visual_ft,
                audio_ft=audio_ft
            )

Another processor class that will be useful to have is to process dialogue data. Here we can define a ``GPTDialogueProcessor`` class.
This processor class receives raw annotations and constructs inputs as a concatenation of input sequences (questions, dialogue contexts, and responses) to facilitate application in GPT models. 
Other methods that are specifically defined are ``padding`` (which is used by dataset instances to pad multiple sequence samples) and ``get_attention_mask`` (which creates an attention mask for Transformer attention in GPT models). 

.. code-block:: python 

    SPECIAL_TOKENS_DICT = {'bos_token': "<bos>", 'eos_token': "<eos>", 'additional_special_tokens': ["<speaker1>", "<speaker2>", "<video>", "<cap>"], 'pad_token': "<pad>"}
    ...

    @registry.register_processor("gpt_dialogue")
    class GPTDialogueProcessor(BaseProcessor):
        def __init__(self, max_turns=3, use_caption=True):
            self.max_turns = max_turns 
            self.use_caption = use_caption 
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self.tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT) 
            
        def sample_sequence(self, caption, history, answer):
            bos, eos, speaker1, speaker2, cap = self.tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-2])
            instance = {}
            sequence = [caption] + history + [answer]
            sequence = [s + [eos] for s in sequence] 
    
            instance["input_ids"] = list(chain(*sequence))
            instance["token_type_ids"] = [cap] * len(sequence[0]) + [speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence[1:]) for _ in s]
            instance["labels"] = ([-1]*sum(len(s) for s in sequence[:-1])) + sequence[-1]
            
            assert len(instance["input_ids"])==len(instance["token_type_ids"])
            assert len(instance["token_type_ids"])==len(instance["labels"])
            
            for k,v in instance.items():
                instance[k] = torch.Tensor(v).long() 
            
            return instance 
        
        def padding(self, seq, pad_token=-1):
            if pad_token==-1: pad_token = self.tokenizer.pad_token_id 
            padded_seq = torch.nn.utils.rnn.pad_sequence(seq, batch_first=True, padding_value=pad_token) 
            return padded_seq
        
        def get_attention_mask(self, seq, pad_token=-1):
            if pad_token==-1: pad_token = self.tokenizer.pad_token_id 
            return seq != pad_token
        
        def __call__(self, ann):
            if self.use_caption:
                caption = ' '.join([ann['caption'], ann['summary']])
                caption = self.tokenizer.encode(caption)
            else:
                caption = []
                
            dial_history = []
            for turn in ann['dialog'][-self.max_turns:]:
                dial_history.append(turn['question'])
                dial_history.append(turn['answer'])
            dial_history.append(ann['question'])
            dial_history = [self.tokenizer.encode(t) for t in dial_history]
            
            answer = self.tokenizer.encode(ann['answer'])
            
            item = self.sample_sequence(caption, dial_history, answer)
            
            return item 
    
        @classmethod
        def from_config(cls, cfg=None):
            if cfg is None:
                cfg = OmegaConf.create()
    
            use_caption = cfg.get("use_caption", True)
            max_turns = cfg.get("max_turns", 3)
    
            return cls(max_turns=max_turns, use_caption=use_caption)

Registering New Processors ``lavis.processors.__init__``
**************************************************************

Finally, any new processor must be officially registered as part of the ``lavis.processors`` module. 
For instance, to add processor classes for GPT-based dialogue models, including one for dialogue data ``GPTDialogueProcessor`` and one for video features ``GPTVideoFeatureProcessor``, we can modify the ``__init__.py`` as follows: 

.. code-block:: python

    from lavis.processors.gpt_processors import (
        GPTVideoFeatureProcessor,
        GPTDialogueProcessor,
    )
    
    __all__ = [
        ...
        # GPT
        "GPTVideoFeatureProcessor",
        "GPTDialogueProcessor"
    ]

Assigning Processors 
**************************************************************
From the above example of processor classes, note that we define a ``from_config`` method for each class. 
This method will process a configuration file and pass specific parameters e.g. ``max_turns``, ``visual_ft``, to initialize the processor classes properly. 
To do this, we can assign/ associate the correct registry of processor classes in a configuration file.
For instance, the following should be specified in a configuration file e.g. ``dialogue_avsd_ft.yaml``:

.. code-block:: yaml 

    datasets:
      avsd_dialogue: # name of the dataset builder
        vis_processor:
            train:
              name: "gpt_video_ft" # name of the visual processor for training data
              visual_ft: ["i3d_flow", "i3d_rgb"]  
              audio_ft: ["vggish"]    
            eval:
              name: "gpt_video_ft" # name of the visual processor for evaluation data
              visual_ft: ["i3d_flow", "i3d_rgb"]  
              audio_ft: ["vggish"]   
        text_processor:
            train:
              name: "gpt_dialogue" # name of the textual processor for training data
              max_turns:  3
              use_caption: True 
            eval:
              name: "gpt_dialogue" # name of the textual processor for evaluation data
              max_turns:  3
              use_caption: True 

Subsequently, any processes (e.g. training) should load this configuration file to assign the correct processors.

.. code-block:: sh

    python train.py --cfg-path dialogue_avsd_ft.yaml

Adding New Models ``lavis.models``
####################################

This is a tutorial on adding new models using ``lavis.models`` module.

The LAVIS library includes a standard model module that builds the foundation for many major language-vision models such as `ALBEF <https://arxiv.org/pdf/2107.07651.pdf>`_,
`BLIP <https://arxiv.org/pdf/2201.12086.pdf>`_, `ALPRO <https://arxiv.org/pdf/2112.09583.pdf>`_, and `CLIP <https://arxiv.org/pdf/2103.00020.pdf>`_. 
The ``lavis.models`` module is designed such that any new models can be added and integrated into the LAVIS library, with minimal steps to develop training and testing procedures. 
In this tutorial, we will replicate the steps to add a GPT-style model specifically for `video-grounded dialogue tasks <https://arxiv.org/pdf/1901.09107.pdf>`_. 

Base Model ``lavis.models.base_model``
**************************************************************

Note that any new model definition should inherit the base model class ``BaseModel``:

.. code-block:: python

    from omegaconf import OmegaConf
    
    import numpy as np
    
    import torch
    import torch.nn as nn
    
    from lavis.common.utils import get_abs_path
    
    class BaseModel(nn.Module):
        """Base class for models."""
    
        def __init__(self):
            super().__init__()
    
        def forward_features(self, *args, **kwargs):
            """Similar to *forward* but only return features."""
            raise NotImplementedError
    
        def load_from_pretrained(self, url_or_filename):
            raise NotImplementedError
    
        @classmethod
        def _from_config(cls, cfg=None, model_type="base"):
            if not cfg:
                # useful when building model without a provided configuration file
                cfg = OmegaConf.load(cls.default_config_path(model_type)).model
    
            return cls.from_config(cfg)
    
        @classmethod
        def from_pretrained(cls, model_type="base"):
            """
            Build a pretrained model from the default configuration file, specified by model_type.
            """
            return cls._from_config(cfg=None, model_type=model_type)
    
        @property
        def device(self):
            return list(self.parameters())[0].device
    
        @classmethod
        def default_config_path(cls, model_type="base"):
            assert (
                model_type in cls.PRETRAINED_MODEL_CONFIG_DICT
            ), "Unknown model type {}".format(model_type)
            return get_abs_path(cls.PRETRAINED_MODEL_CONFIG_DICT[model_type])
    
        def before_evaluation(self, **kwargs):
            pass
    
        def show_n_params(self, return_str=True):
            tot = 0
            for p in self.parameters():
                w = 1
                for x in p.shape:
                    w *= x
                tot += w
            if return_str:
                if tot >= 1e6:
                    return "{:.1f}M".format(tot / 1e6)
                else:
                    return "{:.1f}K".format(tot / 1e3)
            else:
                return tot


In this base model, we already declare and standardize many common methods such as ``_from_config`` and ``_from_pretrained``. 
Inheriting this base model class allows us to standardize operations of models across all model classes while still allowing customizations. 
We advise users not to change the implementation of the base model class as this will affect all existing model subclasses.

GPT-style Video-grounded Dialogue Model ``lavis.models.gpt_models.gpt_dialogue``
********************************************************************************

In this step, we can define a new model class, e.g. under ``lavis.models.gpt_models.gpt_dialogue``, for GPT-based dialogue models designed specifically for video-grounded dialogues. 
Note that we assume the model class inherits from the standard model super class ``GPT2LMHeadModel`` from the ``transformers`` `library <https://huggingface.co/docs/transformers/index>`_.
We also enforce model integration to the LAVIS framework through the inheritance of the ``BaseModel`` from the LAVIS library, as the secondary super class.

.. code-block:: python

    import torch
    from lavis.common.registry import registry
    from lavis.models.base_model import BaseModel
    
    from transformers import GPT2Model, GPT2LMHeadModel
    from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
    import math
    import torch
    import torch.nn as nn
    from torch.nn import CrossEntropyLoss, MSELoss
        
    @registry.register_model("gpt_dialogue")
    class GPTDialogue(GPT2LMHeadModel, BaseModel):
        ...
 
Next, we can modify the architecture of the model during model initialization to fit the tasks of interest, i.e. video-grounded dialogues. 
In this case, we want to add additional model parameters for a linear network to transform the video feature representations to the model dimension. 

.. code-block:: python

    class GPTDialogue(GPT2LMHeadModel, BaseModel):

        def __init__(self, config, len_video_ft=4224):
            
            super().__init__(config)
            
            self.video_ff = nn.Linear(len_video_ft, config.n_embd)
       
            # Model parallel
            self.model_parallel = False
            self.device_map = None
    
            # Initialize weights and apply final processing
            self.post_init()
    
Note that for each new model class, we advise redefining the ``from_config`` method which is inherited from the ``BaseModel`` class.
As each model usually has its own unique configurations, redefining the method will ensure the model instances are created properly. 
For instance, ``GPTDialogue`` requires an additional parameter of video feature length (``len_video_ft``) which should be part of the model initialization procedure. 
Another additional parameter is the number of tokens/words (as we include additional special tokens in the vocabulary for dialogue tasks). 

.. code-block:: python

    class GPTDialogue(GPT2LMHeadModel, BaseModel):
        ...
        @classmethod
        def from_config(cls, cfg):
            model = cls.from_pretrained('gpt2', len_video_ft=cfg['len_video_ft']) 
            model.resize_token_embeddings(cfg['len_tokenizer'])
            return model

Other basic methods should also be defined explicitly in the new model class, including the ``forward`` function. 
For instance, in GPT models for video-grounded dialogue tasks, we want the forward operation also includes the transformation and integration of video features before passing the representations to the Transformer layers. 

.. code-block:: python

    class GPTDialogue(GPT2LMHeadModel, BaseModel):
        ...

        def forward(self, samples, 
                    past_key_values=None,
                    position_ids=None,
                    head_mask=None,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    use_cache=None,
                    output_attentions=None,
                    output_hidden_states=None,
                    return_dict=None):        
                
                input_embs = self.transformer.wte(samples['input_ids'])
                video_embs = self.video_ff(samples['video_fts'])
                input_embs = torch.cat([video_embs, input_embs], dim=1)
                        
                transformer_outputs = self.transformer(
                    attention_mask=samples['attn_mask'],
                    token_type_ids=samples['token_type_ids'],
                    inputs_embeds=input_embs,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
                hidden_states = transformer_outputs[0]
            
                lm_logits = self.lm_head(hidden_states)
                ...

Registering New Model ``lavis.models.__init__``
********************************************************************************

Any new model must be officially registered as part of the ``lavis.models`` module. 
For instance, to add a model class for GPT-based dialogue models, we can modify the ``__init__.py`` as follows:

.. code-block:: python

    from lavis.models.gpt_models.gpt_dialogue import GPTDialogue
    
    __all__ = [
        ...
        "GPTDialogue"
    ]

Assigning Model
********************************************************************************

From the above example of a model class, note that we define a ``from_config method`` for the new model class. 
This method will process a configuration file and pass specific parameters to initialize the model classes properly. 
To do this, we can assign/ associate the correct registry of model classes in a configuration file. 
For instance, the following should be specified in a configuration file e.g. ``dialogue_avsd_ft.yaml``:

.. code-block:: yaml

    model:
      arch: gpt_dialogue # name of the model 
      model_type: base


Subsequently, any processes (e.g. training) should load this configuration file to assign the correct model.

.. code-block:: sh

    python train.py --cfg-path dialogue_avsd_ft.yaml

Note that to simplify the model configuration, we only enable two main parameters here: ``arch`` and ``model_type``. ``arch`` refers to the model class registry, and ``model_type`` is the corresponding model type under this model family.
For instance, with ``gpt_dialogue``, we have a model ``base`` which has its own configuration in a separate configuration file e.g. ``gpt_dialogue_base.yaml``:

.. code-block:: yaml

    model:
      arch: gpt_dialogue
      len_tokenizer: 50264 # 50257 tokens from gpt2 default tokenizer + additional special tokens       
      len_video_ft: 4224 # i3d_rgb: 2048 i3d_flow: 2048 vggish: 128 

We can pass load this configuration and pass the parameters to the above ``from_config`` method to initialize the model accordingly. 
We advise the users to maintain a dictionary that contains default paths to model configurations, in the model class definition. 
By default, the LAVIS framework will search for configurations from each model class defined as ``model.PRETRAINED_MODEL_CONFIG_DICT``.

.. code-block:: python

    class GPTDialogue(GPT2LMHeadModel, BaseModel):
        PRETRAINED_MODEL_CONFIG_DICT = {
                "base": "configs/models/gpt_dialogue_base.yaml"
            }
        ...

Adding New Tasks ``lavis.tasks``
####################################

This is a tutorial on adding new machine learning tasks using ``lavis.tasks`` module.

The LAVIS library includes a standard task module that centralizes the model training and evaluation procedure of machine learning tasks. 
The ``lavis.tasks`` module is designed such that any new tasks can be added and integrated, catering to any customization in the training and testing procedures. 
In this tutorial, we will replicate the steps to add a new task into LAVIS for the `video-grounded dialogue tasks <https://arxiv.org/pdf/1901.09107.pdf>`_. 

Base Task ``lavis.tasks.base_task``
********************************************************************************

Note that any new model definition should inherit the base task class ``BaseTask``:

.. code-block:: python

    import logging
    import os
    
    import torch.distributed as dist
    from lavis.common.dist_utils import get_rank, get_world_size, is_main_process
    from lavis.common.logger import MetricLogger, SmoothedValue
    from lavis.common.registry import registry
    from lavis.datasets.data_utils import prepare_sample
    
    class BaseTask:
        def __init__(self, **kwargs):
            super().__init__()
    
            self.inst_id_key = "instance_id"
    
        @classmethod
        def setup_task(cls, **kwargs):
            return cls()
    
        def build_model(self, cfg):
            model_config = cfg.model_cfg
    
            model_cls = registry.get_model_class(model_config.arch)
            return model_cls.from_config(model_config)
    
        def build_datasets(self, cfg):
            """
            Build a dictionary of datasets, keyed by split 'train', 'valid', 'test'.
            Download dataset and annotations automatically if not exist.
    
            Args:
                cfg (common.config.Config): _description_
    
            Returns:
                dict: Dictionary of torch.utils.data.Dataset objects by split.
            """
    
            datasets = dict()
    
            datasets_config = cfg.datasets_cfg
    
            assert len(datasets_config) > 0, "At least one dataset has to be specified."
    
            for name in datasets_config:
                dataset_config = datasets_config[name]
    
                builder = registry.get_builder_class(name)(dataset_config)
                dataset = builder.build_datasets()
    
                datasets[name] = dataset
    
            return datasets
    
        def train_step(self, model, samples):
            loss = model(samples)["loss"]
            return loss
    
        ...

In this base task, we already declare and standardize many common methods such as ``train_step``, ``build_model``, and ``build_datasets``. 
Inheriting this base task class allows us to standardize operations of tasks across all task classes.
We recommend users not change the implementation of the base task class as this will have an impact on all existing task subclasses.

Dialogue Task ``lavis.tasks.dialogue``
********************************************************************************

In this step, we can define a new task class, e.g. under ``lavis.tasks.dialogue``, for video-grounded dialogues.
For instance, we define a new task class ``DialogueTask`` that inherits the super task class ``BaseTask``.

.. code-block:: python

    import json
    import os
    
    from lavis.common.dist_utils import main_process
    from lavis.common.logger import MetricLogger
    from lavis.common.registry import registry
    from lavis.tasks.base_task import BaseTask
    from lavis.datasets.data_utils import prepare_sample
    
    import numpy as np 
    
    @registry.register_task("dialogue")
    class DialogueTask(BaseTask):
        def __init__(self, num_beams, max_len, min_len, evaluate, report_metric=True):
            super().__init__()
    
            self.num_beams = num_beams
            self.max_len = max_len
            self.min_len = min_len
            self.evaluate = evaluate
    
            self.report_metric = report_metric
    
        @classmethod
        def setup_task(cls, cfg):
            run_cfg = cfg.run_cfg
    
            num_beams = run_cfg.num_beams
            max_len = run_cfg.max_len
            min_len = run_cfg.min_len
            evaluate = run_cfg.evaluate
    
            report_metric = run_cfg.get("report_metric", True)
    
            return cls(
                num_beams=num_beams,
                max_len=max_len,
                min_len=min_len,
                evaluate=evaluate,
                report_metric=report_metric,
            )
    
        def valid_step(self, model, samples):
            results = []        
            loss = model(samples)["loss"].item() 
            
            return [loss] 
        ...

Note that for any new task, we advise the users to review carefully the functions implemented within ``BaseTask`` and consider which methods should be modified. 
For instance, the base task class already contains a standard implementation of model training steps that are common among machine learning steps. 
Some major methods we want to emphasize and should be customized by each task are the ``valid_step`` and ``evaluation``. 
These operations were not fully implemented in the base task class due to the differences in evaluation procedures among many machine learning tasks. 
Another method that should be considered is the ``setup_task`` method. 
This method will receive configurations that set task-specific parameters to initialize any task instance.

Registering New Task ``lavis.tasks.__init__`` 
********************************************************************************

Any new task must be officially registered as part of the ``lavis.tasks`` module. For instance, to add a new task for video-grounded dialogues, we can modify the ``__init__.py`` as follows:

.. code-block:: python

    from lavis.tasks.dialogue import DialogueTask
    
    ...
    __all__ = [
        ...
        "DialogueTask"
    ]

Assigning Task 
***************

From the above example of task class, note that we define a ``setup_task`` method for each task class. 
This method will process a configuration file and pass specific parameters e.g. ``num_beams`` (for beam search generative tasks during the inference stage), to initialize the task classes properly. 
To assign and associate any task, we need to specify the correct registry of task classes in a configuration file. 
For instance, the following should be specified in a configuration file e.g. ``dialogue_avsd_ft.yaml``:

.. code-block:: yaml

    run:
      task: dialogue # name of the task 
      
      # optimizer
      ...
    
      max_len: 20
      min_len: 5
      num_beams: 3    
      ...
    
Subsequently, any processes (e.g. training) should load this configuration file to assign the correct task.

.. code-block:: sh

    python train.py --cfg-path dialogue_avsd_ft.yaml