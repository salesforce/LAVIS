(Image credit: "https://arxiv.org/pdf/1504.00325.pdf")

# Microsoft COCO Caption

## Description
[Microsoft COCO Captions dataset](https://github.com/tylin/coco-caption) contains over one and a half million captions describing over 330,000 images. For the training and validation images, five independent human generated captions are be provided for each image.

## Task
**Image captioning** is the task of describing the content of an image in words. This task lies at the intersection of computer vision and natural language processing. Most image captioning systems use an encoder-decoder framework, where an input image is encoded into an intermediate representation of the information in the image, and then decoded into a descriptive text sequence. The most popular benchmarks are nocaps and COCO, and models are typically evaluated according to a BLEU or CIDER metric (copied from [link](https://paperswithcode.com/task/image-captioning)).

## Leaderboard
| Rank. |  Model  | BLEU-4 | CIDEr | METEOR | SPICE |                                                                    Resources                                                                     |
| ----- | :-----: | :----: | :---: | :----: | :---: | :----------------------------------------------------------------------------------------------------------------------------------------------: |
| 1     |   OFA   |  44.9  | 154.9 |  32.5  | 26.6  |                                [paper](https://arxiv.org/abs/2202.03052), [code](https://github.com/OFA-Sys/OFA)                                 |
| 2     |  LEMON  |  42.6  | 145.5 |  31.4  | 25.5  |                                                                    [paper]()                                                                     |
| 3     | SimVLM  |  40.6  | 143.3 |  33.7  | 25.4  |                                                [paper](https://openreview.net/pdf?id=GUrhfTuf_3)                                                 |
| 4     |  VinVL  |  41.0  | 140.9 |  31.1  | 25.2  |                           [paper](https://arxiv.org/pdf/2101.00529v2.pdf), [code](https://github.com/microsoft/Oscar)                            |
| 5     |  OSCAR  |  40.7  | 140.0 |  30.6  | 24.5  |                           [paper](https://arxiv.org/pdf/2004.06165v5.pdf), [code](https://github.com/microsoft/Oscar)                            |
| 6     |  BLIP   |  40.4  | 136.7 |  31.4  | 24.3  | [paper](https://arxiv.org/pdf/2201.12086.pdf), [code](https://github.com/salesforce/BLIP), [demo](https://huggingface.co/spaces/Salesforce/BLIP) |
| 7     |   M^2   |  39.1  | 131.2 |  29.2  | 22.6  |                 [paper](https://arxiv.org/pdf/1912.08226v2.pdf), [code](https://github.com/aimagelab/meshed-memory-transformer)                  |
| 8     |  BUTD   |  36.5  | 113.5 |  27.0  | 20.3  |               [paper](https://arxiv.org/abs/1707.07998?context=cs), [code](https://github.com/peteanderson80/bottom-up-attention)                |
| 9     | ClipCap |  32.2  | 108.4 |  27.1  | 20.1  |                     [paper](https://arxiv.org/pdf/2111.09734v1.pdf), [code](https://github.com/rmokady/clip_prefix_caption)                      |

## References
"Microsoft COCO Captions: Data Collection and Evaluation Server", Xinlei Chen, Hao Fang, Tsung-Yi Lin, Ramakrishna Vedantam, Saurabh Gupta, Piotr Dollar, C. Lawrence Zitnick
