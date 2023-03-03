.. _config:

Training Models on Task Datasets (Commands and Configurations) 
#################################################################

LAVIS provides scripts to pre-train and finetune supported models on standard language-vision tasks, stored at ``lavis/run_scripts/``. 
To replicate the experiments, just run these bash scripts. For example, to train BLIP model on the image-text retrieval task with MSCOCO dataset, we can run

.. code-block::

    bash run_scripts/blip/train/train_retrieval_coco.sh

Inside the scripts, we can see 

.. code-block:: bash

    python -m torch.distributed.run --nproc_per_node=8 train.py --cfg-path lavis/projects/blip/train/retrieval_coco_ft.yaml

where we start a pytorch distributed training on 8 GPUs (you may change according to your own hardware setup). The ``--cfg-path`` specifys a `runtime configuration file`, specifying
the task, model, dataset and training recipes. 

Available options and their descriptions are as below.

.. LAVIS executes training and evaluation based on arguments specified in the configuration files. The default model and dataset configurations are defined in ``lavis/configs``. The task-specific configurations are defined in ``lavis/projects``. Task-specific configurations have higher priority over the default configurations.

.. The following tables provide explanations for the arguments in the configuration files.

.. list-table::
   :widths: 30 40
   :header-rows: 1

   * - Model Configurations
     - Functionalities
   * - arch
     - | name of the model from the model zoo
       | default: task-dependent
   * - model_type
     - | the type of the model (e.g., base)
       | default: task-dependent
   * - load_pretrained
     - | load pretrained weights
       | default: True (for finetuning task) | False (for pretraining task) 
   * - load_finetuned
     - | load task-specific finetuned weights
       | default: False (for finetuning task) | True (for evaluation) 
   * - pretrained 
     - | URL or local path which stores the pretrained model, defined in the default model configuration file
       | default: task-dependent 
   * - finetuned
     - | URL or local path which stores the finetuned model, defined in the default model configuration file
       | default: task-dependent

.. list-table::
   :widths: 30 50
   :header-rows: 1

   * - Dataset Configurations
     - Functionalities
   * - vis_processor
     - | pre-processing of visual input
       | default: task-dependent
   * - text_processor
     - | pre-processing of text input
       | default: task-dependent
   * - build_info
     - | dataset information including the storage location, defined in the default dataset configuration file
       | default: task-dependent

.. list-table::
   :widths: 30 50
   :header-rows: 1

   * - Runtime Configurations
     - Functionalities
   * - task
     - | name of the task
       | default: task-dependent
   * - lr_sched
     - | learning rate schedular
       | default: linear_warmup_cosine_lr
   * - init_lr
     - | initial learning rate (after warmup)
       | default: task-dependent
   * - min_lr
     - | final learning rate after decay
       | default: task-dependent
   * - warmup_lr
     - | starting learning rate for warmup
       | default: init_lr (no warmup)
   * - lr_decay_rate
     - | learning rate decay per epoch for step_lr_shedule
       | default: 0.9
   * - warmup_steps
     - | number of steps for learning rate warmup
       | default: 0
   * - max_epoch
     - | total number of training epochs
       | default: task-dependent
   * - weight_decay
     - | weight decay coefficient for the optimizer
       | default: 0.05
   * - batch_size_train
     - | batch size during training
       | default: task-dependent
   * - batch_size_eval
     - | batch size during evaluation
       | default: task-dependent
   * - seed
     - | pseudo random number generator seed
       | default: 42
   * - output_dir
     - | directory to store logs, results and checkpoints
       | default: task-dependent
   * - resume_ckpt_path
     - | path of the checkpoint to resume training from
       | default: None
   * - evaluate
     - | only perform evaluation without training
       | default: False
   * - train_splits
     - | dataset splits used for training
       | default: ["train"]
   * - valid_splits
     - | dataset splits used for validation
       | default: ["val"]
   * - test
     - | dataset splits used for test
       | default: ["test"]
   * - device
     - | use cpu or gpu (cuda)
       | default: cuda
   * - world_size
     - | number of processes participating in the job
       | default: 1
   * - dist_url
     - | URL specifying how to initialize the process group
       | default: "env://"
   * - distributed
     - | use distributed training
       | default: True
   * - amp
     - | use automatic mixed precision training
       | default: False

.. list-table::
   :widths: 40 50
   :header-rows: 1

   * - Text Generation Configurations
     - Functionalities
   * - max_len
     - | maximum number of text tokens to generate
       | default: 20 (for image captioning)
   * - min_len
     - | minimum number of text tokens to generate
       | default: 5 (for image captioning)
   * - num_beams
     - | number of beams to perform beam search
       | default: 3

.. list-table::
   :widths: 40 50
   :header-rows: 1

   * - Multimodal Retrieval Configurations
     - Functionalities
   * - negative_all_rank
     - | collect negatives from all processes for the image-text matching loss
       | default: True (for coco)
   * - k_test
     - | number of retrieval candidates ranked from contrastive similarity
       | default: 256 (for coco)
