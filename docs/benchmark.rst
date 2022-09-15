Benchmark
############

We provide scripts for evaluating and training models on task datasets. The following benchmark results are included for reference.


ALBEF
*******
.. list-table::
   :widths: 30 80 20

   * - **Pretraining**
     - COCO (`download <https://github.com/salesforce/LAVIS/blob/main/lavis/datasets/download_scripts/download_coco.py>`__)
     - `script <https://github.com/salesforce/LAVIS/blob/main/run_scripts/albef/train/pretrain.sh>`__
   * -
     - Visual Genome (`download <https://github.com/salesforce/LAVIS/blob/main/lavis/datasets/download_scripts/download_vg.py>`__)
     -
   * -
     - SBU (`download <https://github.com/salesforce/LAVIS/blob/main/lavis/datasets/download_scripts/download_sbu.py>`__)
     -
   * -
     - CC3M (`download <https://github.com/salesforce/LAVIS/blob/main/lavis/datasets/download_scripts/DownloadConceptualCaptions/download_data_cc3m.py>`__)
     -
   * -
     - CC12M (`download <https://github.com/salesforce/LAVIS/blob/main/lavis/datasets/download_scripts/DownloadConceptualCaptions/download_data_cc12m.py>`__)
     -

.. list-table::
   :widths: 30 40 20 20 20 30 30
   :header-rows: 1

   * -
     - **Retrieval**
     - **R1**
     - **R5**
     - **R10**
     - **Training**
     - **Evaluation**
   * - TR
     - COCO (`download <https://github.com/salesforce/LAVIS/blob/main/lavis/datasets/download_scripts/download_coco.py>`__)
     - 77.6
     - 94.1
     - 97.2
     - `script <https://github.com/salesforce/LAVIS/blob/main/run_scripts/albef/train/train_coco_retrieval_albef.sh>`__
     - `script <https://github.com/salesforce/LAVIS/blob/main/run_scripts/albef/eval/eval_coco_retrieval.sh>`__
   * - IR
     - COCO (`download <https://github.com/salesforce/LAVIS/blob/main/lavis/datasets/download_scripts/download_coco.py>`__)
     - 61.0
     - 84.5
     - 90.7
     - `script <https://github.com/salesforce/LAVIS/blob/main/run_scripts/albef/train/train_coco_retrieval_albef.sh>`__
     - `script <https://github.com/salesforce/LAVIS/blob/main/run_scripts/albef/eval/eval_coco_retrieval.sh>`__
   * - TR
     - Flickr30k (`download <https://github.com/salesforce/LAVIS/blob/main/lavis/datasets/download_scripts/download_flickr.py>`__)
     - 77.6
     - 94.1
     - 97.2
     - `script <https://github.com/salesforce/LAVIS/blob/main/run_scripts/albef/train/train_flickr30k_retrieval_albef.sh>`__
     - `script <https://github.com/salesforce/LAVIS/blob/main/run_scripts/albef/eval/eval_flickr30k_retrieval.sh>`__
   * - IR
     - Flickr30k (`download <https://github.com/salesforce/LAVIS/blob/main/lavis/datasets/download_scripts/download_flickr.py>`__)
     - 61.0
     - 84.5
     - 90.7
     - `script <https://github.com/salesforce/LAVIS/blob/main/run_scripts/albef/train/train_flickr30k_retrieval_albef.sh>`__
     - `script <https://github.com/salesforce/LAVIS/blob/main/run_scripts/albef/eval/eval_flickr30k_retrieval.sh>`__


.. list-table::
   :widths: 20 20 20 20 20
   :header-rows: 1

   * - **VQA**
     - **test-dev**
     - **test-std/test**
     - **Training**
     - **Evaluation**
   * - VQAv2 (`download <https://github.com/salesforce/LAVIS/blob/main/lavis/datasets/download_scripts/download_coco.py>`__)
     - 76.35
     - 76.54
     - `script <https://github.com/salesforce/LAVIS/blob/main/run_scripts/albef/train/train_vqa_albef.sh>`__
     - `script <https://github.com/salesforce/LAVIS/blob/main/run_scripts/albef/eval/test_albef_vqa.sh>`__
   * - OKVQA (`download <https://github.com/salesforce/LAVIS/blob/main/lavis/datasets/download_scripts/download_coco.py>`__)
     - NA
     - 54.7 
     - `script <https://github.com/salesforce/LAVIS/blob/main/run_scripts/albef/train/train_okvqa_albef.sh>`__
     - NA
   * - AOKVQA (`download <https://github.com/salesforce/LAVIS/blob/main/lavis/datasets/download_scripts/download_coco.py>`__)
     - 54.5
     - NA
     - `script <https://github.com/salesforce/LAVIS/blob/main/run_scripts/albef/train/train_aokvqa_albef.sh>`__
     - NA

  
.. list-table::
   :widths: 20 20 20 20 20
   :header-rows: 1

   * - **Multimodal Classification**
     - **val**
     - **test**
     - **Training**
     - **Evaluation**
   * - SNLI-VE (`download <https://github.com/salesforce/LAVIS/blob/main/lavis/datasets/download_scripts/download_coco.py>`__)
     - 80.60
     - 81.04
     - `script <https://github.com/salesforce/LAVIS/blob/main/run_scripts/albef/train/train_ve_albef.sh>`__
     - `script <https://github.com/salesforce/LAVIS/blob/main/run_scripts/albef/eval/eval_albef_ve.sh>`__
   * - NLVR2 (`download <https://github.com/salesforce/LAVIS/blob/main/lavis/datasets/download_scripts/download_coco.py>`__)
     - 82.47 
     - 82.91 
     - `script <https://github.com/salesforce/LAVIS/blob/main/run_scripts/albef/train/train_nlvr_albef.sh>`__
     - `script <https://github.com/salesforce/LAVIS/blob/main/run_scripts/albef/eval/eval_albef_nlvr.sh>`__
  
BLIP
*******
.. list-table::
   :widths: 30 80 20

   * - **Pretraining (14M)**
     - COCO (`download <https://github.com/salesforce/LAVIS/blob/main/lavis/datasets/download_scripts/download_coco.py>`__)
     - `script <https://github.com/salesforce/LAVIS/blob/main/run_scripts/blip/train/pretrain.sh>`__
   * -
     - Visual Genome (`download <https://github.com/salesforce/LAVIS/blob/main/lavis/datasets/download_scripts/download_vg.py>`__)
     -
   * -
     - SBU (`download <https://github.com/salesforce/LAVIS/blob/main/lavis/datasets/download_scripts/download_sbu.py>`__)
     -
   * -
     - CC3M (`download <https://github.com/salesforce/LAVIS/blob/main/lavis/datasets/download_scripts/DownloadConceptualCaptions/download_data_cc3m.py>`__)
     -
   * -
     - CC12M (`download <https://github.com/salesforce/LAVIS/blob/main/lavis/datasets/download_scripts/DownloadConceptualCaptions/download_data_cc12m.py>`__)
     -

.. list-table::
   :widths: 30 40 20 20 20 30 30
   :header-rows: 1

   * - **Tasks**
     - **Retrieval**
     - **R1**
     - **R5**
     - **R10**
     - **Training**
     - **Evaluation**
   * - TR
     - COCO (`download <https://github.com/salesforce/LAVIS/blob/main/lavis/datasets/download_scripts/download_coco.py>`__)
     - 82.0
     - 95.8
     - 98.1
     - `script <https://github.com/salesforce/LAVIS/blob/main/run_scripts/blip/train/train_retrieval_coco.sh>`__
     - `script <https://github.com/salesforce/LAVIS/blob/main/run_scripts/blip/eval/eval_ret_coco.sh>`__
   * - IR
     - COCO (`download <https://github.com/salesforce/LAVIS/blob/main/lavis/datasets/download_scripts/download_coco.py>`__)
     - 64.5
     - 86.0
     - 91.7
     - `script <https://github.com/salesforce/LAVIS/blob/main/run_scripts/blip/train/train_retrieval_coco.sh>`__
     - `script <https://github.com/salesforce/LAVIS/blob/main/run_scripts/blip/eval/eval_ret_coco.sh>`__
   * - TR
     - Flickr30k (`download <https://github.com/salesforce/LAVIS/blob/main/lavis/datasets/download_scripts/download_flickr.py>`__)
     - 96.9
     - 99.9
     - 100.0
     - `script <https://github.com/salesforce/LAVIS/blob/main/run_scripts/blip/train/train_retrieval_flickr.sh>`__
     - `script <https://github.com/salesforce/LAVIS/blob/main/run_scripts/blip/eval/eval_ret_flickr.sh>`__
   * - IR
     - Flickr30k (`download <https://github.com/salesforce/LAVIS/blob/main/lavis/datasets/download_scripts/download_flickr.py>`__)
     - 87.5
     - 97.6
     - 98.9
     - `script <https://github.com/salesforce/LAVIS/blob/main/run_scripts/blip/train/train_retrieval_flickr.sh>`__
     - `script <https://github.com/salesforce/LAVIS/blob/main/run_scripts/blip/eval/eval_ret_flickr.sh>`__


.. list-table::
   :widths: 20 20 20 20 20
   :header-rows: 1

   * - **VQA**
     - **test-dev**
     - **test-std/test**
     - **Training**
     - **Evaluation**
   * - VQAv2 (`download <https://github.com/salesforce/LAVIS/blob/main/lavis/datasets/download_scripts/download_coco.py>`__)
     - 78.23
     - 78.29
     - `script <https://github.com/salesforce/LAVIS/blob/main/run_scripts/albef/train/train_vqa_albef.sh>`__
     - `script <https://github.com/salesforce/LAVIS/blob/main/run_scripts/albef/eval/test_albef_vqa.sh>`__
   * - OKVQA (`download <https://github.com/salesforce/LAVIS/blob/main/lavis/datasets/download_scripts/download_coco.py>`__)
     - NA
     - 55.4 
     - `script <https://github.com/salesforce/LAVIS/blob/main/run_scripts/blip/train/train_okvqa.sh>`__
     - `script <https://github.com/salesforce/LAVIS/blob/main/run_scripts/blip/eval/eval_okvqa.sh>`__
   * - AOKVQA (`download <https://github.com/salesforce/LAVIS/blob/main/lavis/datasets/download_scripts/download_coco.py>`__)
     - 56.2
     - 50.1 
     - `script <https://github.com/salesforce/LAVIS/blob/main/run_scripts/blip/train/train_aokvqa.sh>`__
     - `script <https://github.com/salesforce/LAVIS/blob/main/run_scripts/blip/eval/eval_aokvqa.sh>`__


.. list-table::
   :widths: 20 20 20 20 20 20
   :header-rows: 1

   * - **Image Captioning**
     - **BLEU@4**
     - **CIDEr**
     - **SPICE**
     - **Training**
     - **Evaluation**
   * - COCO (`download <https://github.com/salesforce/LAVIS/blob/main/lavis/datasets/download_scripts/download_coco.py>`__)
     - 39.9
     - 133.5
     - 23.7
     - `script <https://github.com/salesforce/LAVIS/blob/main/run_scripts/blip/train/train_caption_coco.sh>`__
     - `script <https://github.com/salesforce/LAVIS/blob/main/run_scripts/blip/eval/eval_coco_cap.sh>`__
   * - NoCaps (`download <https://github.com/salesforce/LAVIS/blob/main/lavis/datasets/download_scripts/download_nocaps.py>`__)
     - 31.9
     - 109.1
     - 14.7
     - NA
     - `script <https://github.com/salesforce/LAVIS/blob/main/run_scripts/blip/eval/eval_nocaps.sh>`__


.. list-table::
   :widths: 20 20 20 20 20
   :header-rows: 1

   * - **Multimodal Classification**
     - **val**
     - **test**
     - **Training**
     - **Evaluation**
   * - NLVR2 (`download <https://github.com/salesforce/LAVIS/blob/main/lavis/datasets/download_scripts/download_coco.py>`__)
     - 82.48
     - 83.25
     - `script <https://github.com/salesforce/LAVIS/blob/main/run_scripts/blip/train/train_nlvr.sh>`__
     - `script <https://github.com/salesforce/LAVIS/blob/main/run_scripts/blip/eval/eval_nlvr.sh>`__

CLIP
*******
.. list-table::
   :widths: 30 40 20 20 20 30
   :header-rows: 1

   * - **Tasks**
     - **Retrieval (Zero-shot)**
     - **R1**
     - **R5**
     - **R10**
     - **Evaluation**
   * - TR
     - COCO (`download <https://github.com/salesforce/LAVIS/blob/main/lavis/datasets/download_scripts/download_coco.py>`__)
     - 57.2
     - 80.5
     - 87.8
     - `script <https://github.com/salesforce/LAVIS/blob/main/run_scripts/clip/eval/eval_clip_ret_coco.sh>`__
   * - IR
     - COCO (`download <https://github.com/salesforce/LAVIS/blob/main/lavis/datasets/download_scripts/download_coco.py>`__)
     - 36.5
     - 60.8
     - 71.0
     - `script <https://github.com/salesforce/LAVIS/blob/main/run_scripts/clip/eval/eval_clip_ret_coco.sh>`__
   * - TR
     - Flickr30k (`download <https://github.com/salesforce/LAVIS/blob/main/lavis/datasets/download_scripts/download_flickr.py>`__)
     - 86.5
     - 98.0
     - 99.1
     - `script <https://github.com/salesforce/LAVIS/blob/main/run_scripts/clip/eval/eval_clip_ret_flickr.sh>`__
   * - IR
     - Flickr30k (`download <https://github.com/salesforce/LAVIS/blob/main/lavis/datasets/download_scripts/download_flickr.py>`__)
     - 67.0
     - 88.9
     - 93.3
     - `script <https://github.com/salesforce/LAVIS/blob/main/run_scripts/clip/eval/eval_clip_ret_flickr.sh>`__

.. list-table::
   :widths: 20 20 20
   :header-rows: 1

   * - **Multimodal Classification**
     - **val**
     - **Evaluation**
   * - ImageNet 
     - 76.5 
     - `script <https://github.com/salesforce/LAVIS/blob/main/run_scripts/clip/eval/eval_clip_zs_imnet.sh>`__


ALPRO
*******
.. list-table::
   :widths: 30 40 20 20 20 20 30
   :header-rows: 1

   * - **Tasks**
     - **Retrieval**
     - **R1**
     - **R5**
     - **R10**
     - **Training**
     - **Evaluation**
   * - TR
     - MSRVTT (`download <https://github.com/salesforce/LAVIS/blob/main/lavis/datasets/download_scripts/download_msrvtt.py>`__)
     - 33.2
     - 60.5 
     - 71.7 
     - `script <https://github.com/salesforce/LAVIS/blob/main/run_scripts/alpro/train/train_msrvtt_ret.sh>`__
     - `script <https://github.com/salesforce/LAVIS/blob/main/run_scripts/alpro/eval/eval_msrvtt_ret.sh>`__
   * - VR
     - MSRVTT (`download <https://github.com/salesforce/LAVIS/blob/main/lavis/datasets/download_scripts/download_msrvtt.py>`__)
     - 33.8
     - 61.4
     - 72.7
     - `script <https://github.com/salesforce/LAVIS/blob/main/run_scripts/alpro/train/train_msrvtt_ret.sh>`__
     - `script <https://github.com/salesforce/LAVIS/blob/main/run_scripts/alpro/eval/eval_msrvtt_ret.sh>`__
   * - TR
     - DiDeMo (`download <https://github.com/salesforce/LAVIS/blob/main/lavis/datasets/download_scripts/download_didemo.py>`__)
     - 38.8 
     - 66.4
     - 76.8
     - `script <https://github.com/salesforce/LAVIS/blob/main/run_scripts/alpro/train/train_didemo_ret.sh>`__
     - `script <https://github.com/salesforce/LAVIS/blob/main/run_scripts/alpro/eval/eval_didemo_ret.sh>`__
   * - VR
     - DiDeMo (`download <https://github.com/salesforce/LAVIS/blob/main/lavis/datasets/download_scripts/download_didemo.py>`__)
     - 36.6
     - 67.5
     - 77.9
     - `script <https://github.com/salesforce/LAVIS/blob/main/run_scripts/alpro/train/train_didemo_ret.sh>`__
     - `script <https://github.com/salesforce/LAVIS/blob/main/run_scripts/alpro/eval/eval_didemo_ret.sh>`__

.. list-table::
   :widths: 20 20 20 20
   :header-rows: 1

   * - **Video QA**
     - **test**
     - **Training**
     - **Evaluation**
   * - MSRVTT 
     - 42.1 
     - `script <https://github.com/salesforce/LAVIS/blob/main/run_scripts/alpro/train/train_msrvtt_qa.sh>`__
     - `script <https://github.com/salesforce/LAVIS/blob/main/run_scripts/alpro/eval/eval_msrvtt_qa.sh>`__
   * - MSVD 
     - 46.0 
     - `script <https://github.com/salesforce/LAVIS/blob/main/run_scripts/alpro/train/train_msvd_qa.sh>`__ 
     - `script <https://github.com/salesforce/LAVIS/blob/main/run_scripts/alpro/eval/eval_msvd_qa.sh>`__