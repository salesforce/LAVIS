![Samples from the COCO Caption dataset (Image credit: "https://arxiv.org/pdf/1504.00325.pdf").](imgs/nocaps.png)

# Nocaps

## Description

our benchmark consists of 166,100 human-generated captions describing 15,100 images from the Open Images validation and test sets. The associated training data consists of COCO image-caption pairs, plus Open Images image-level labels and object bounding boxes. Since Open Images contains many more classes than COCO, nearly 400 object classes seen in test images have no or very few associated training captions (hence, nocaps).


## Task: Novel object captioning

(from https://nocaps.org/)

Image captioning models have achieved impressive results on datasets containing limited visual concepts and large amounts of paired image-caption training data. However, if these models are to ever function in the wild, a much larger variety of visual concepts must be learned, ideally from less supervision. To encourage the development of image captioning models that can learn visual concepts from alternative data sources, such as object detection datasets, we present the first large-scale benchmark for this task. Dubbed nocaps, for novel object captioning at scale


## Metrics
Models are typically evaluated according to a [CIDEr](https://aclanthology.org/P02-1040/) or [SPICE](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Vedantam_CIDEr_Consensus-Based_Image_2015_CVPR_paper.pdf) metric.

## Leaderboard

(Ranked by CIDEr)

| Rank |  Model  | val. CIDEr | val. SPICE |          test CIDEr | test SPICE |                                                          Resources                                                                     |
| ---- | :-----: | :----: | :---: | :----------------------------------------------------------------------------------------------------------------------------------------------: | :---:|:---:|
| 1    |  CoCa  |   122.4   |  15.5  | 120.6 | 15.5| [paper](https://arxiv.org/pdf/2205.01917.pdf) |
| 2    |  LEMON  |  117.3 | 15.0  |114.3 | 14.9           |                                                         [paper]()                                                                     |
| 3    |  BLIP   |  113.2 | 14.8 |  -  | -  | [paper](https://arxiv.org/pdf/2201.12086.pdf), [code](https://github.com/salesforce/BLIP), [demo](https://huggingface.co/spaces/Salesforce/BLIP) |
| 4    | SimVLM  |  112.2  | - | 110.3  | 14.5  |                                                [paper](https://openreview.net/pdf?id=GUrhfTuf_3)                                                 |

| 4    |  VinVL  |  41.0  | 140.9 |  103.7  | 14.4  |                           [paper](https://arxiv.org/pdf/2101.00529v2.pdf), [code](https://github.com/microsoft/Oscar)                            |
| 5    |  OSCAR  |  40.7  | 140.0 |  30.6  | 24.5  |                           [paper](https://arxiv.org/pdf/2004.06165v5.pdf), [code](https://github.com/microsoft/Oscar)                            |
| 6    |  BLIP   |  40.4  | 136.7 |  31.4  | 24.3  | [paper](https://arxiv.org/pdf/2201.12086.pdf), [code](https://github.com/salesforce/BLIP), [demo](https://huggingface.co/spaces/Salesforce/BLIP) |
| 7    |   M^2   |  39.1  | 131.2 |  29.2  | 22.6  |                 [paper](https://arxiv.org/pdf/1912.08226v2.pdf), [code](https://github.com/aimagelab/meshed-memory-transformer)                  |
| 8    |  BUTD   |  36.5  | 113.5 |  27.0  | 20.3  |               [paper](https://arxiv.org/abs/1707.07998?context=cs), [code](https://github.com/peteanderson80/bottom-up-attention)                |
| 9    | ClipCap |  32.2  | 108.4 |  27.1  | 20.1  |                     [paper](https://arxiv.org/pdf/2111.09734v1.pdf), [code](https://github.com/rmokady/clip_prefix_caption)                      |

## References
"Microsoft COCO Captions: Data Collection and Evaluation Server", Xinlei Chen, Hao Fang, Tsung-Yi Lin, Ramakrishna Vedantam, Saurabh Gupta, Piotr Dollar, C. Lawrence Zitnick
