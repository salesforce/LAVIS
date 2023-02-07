Evaluating Pre-trained Models on Task Datasets
###############################################
LAVIS provides pre-trained and finetuned model for off-the-shelf evaluation on task dataset. 
Let's now see an example to evaluate BLIP model on the captioning task, using MSCOCO dataset.

.. _prep coco:

Preparing Datasets
******************
First, let's download the dataset. LAVIS provides `automatic downloading scripts` to help prepare 
most of the public dataset, to download MSCOCO dataset, simply run

.. code-block:: bash

    cd lavis/datasets/download_scripts && python download_coco.py

This will put the downloaded dataset at a default cache location ``cache`` used by LAVIS.

If you want to use a different cache location, you can specify it by updating ``cache_root`` in ``lavis/configs/default.yaml``.

If you have a local copy of the dataset, it is recommended to create a symlink from the cache location to the local copy, e.g.

.. code-block:: bash

    ln -s /path/to/local/coco cache/coco

Evaluating pre-trained models
******************************

To evaluate pre-trained model, simply run

.. code-block:: bash

    bash run_scripts/blip/eval/eval_coco_cap.sh

Or to evaluate a large model:

.. code-block:: bash

    bash run_scripts/blip/eval/eval_coco_cap_large.sh
