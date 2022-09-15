Adding Processors
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
