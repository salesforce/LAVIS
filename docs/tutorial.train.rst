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
