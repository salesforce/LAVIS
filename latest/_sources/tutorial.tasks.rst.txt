Adding Tasks
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