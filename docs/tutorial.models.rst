Adding Models
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
