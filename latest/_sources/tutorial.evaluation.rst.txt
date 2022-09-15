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