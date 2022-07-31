![Samples from MSVD-QA dataset.](imgs/msvd_qa.png)(Samples from MSVD-QA dataset, image credit: http://staff.ustc.edu.cn/~hexn/papers/mm17-videoQA.pdf)

# MSVD Dataset (Video Question Answering)

## Description
[MSVD-QA](http://staff.ustc.edu.cn/~hexn/papers/mm17-videoQA.pdf) dataset is based on Microsoft Research Video
Description Corpus (https://www.cs.utexas.edu/users/ml/clamp/videoDescription/) which is used in many video captioning
experiments. The MSVD-QA dataset has a total number of 1,970
video clips and 50,505 question answer pairs.


## Task
Video question answering (VideoQA) is the task where
a video and a natural language question are provided and the model
needs to give the right answer (from [paper](http://staff.ustc.edu.cn/~hexn/papers/mm17-videoQA.pdf)).


## Metrics
Accuracy.

## Leaderboard
(Ranked by accurarcy on test-dev.)
| Rank | Model  | Acc. | Resources |
| ---- | :----: | :-------: | :-------: |
| 1   |  VQA-T  |  46.3 | [paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Yang_Just_Ask_Learning_To_Answer_Questions_From_Millions_of_Narrated_ICCV_2021_paper.pdf), [code](https://github.com/antoyang/just-ask), [demo](http://videoqa.paris.inria.fr/) |
| 2    |  ALPro  |  45.9 |  [paper](https://arxiv.org/abs/2112.09583), [code](https://github.com/salesforce/ALPRO), [blog](https://blog.salesforceairesearch.com/alpro/) |
| 3   |  CoMVT | 42.6 | [paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Seo_Look_Before_You_Speak_Visually_Contextualized_Utterances_CVPR_2021_paper.pdf) |
| 4   |  DualVGR | 39.0 | [paper](https://arxiv.org/pdf/2107.04768v1.pdf) [code](https://github.com/NJUPT-MCC/DualVGR-VideoQA) |
| 5   |  HCRN | 36.1 | [paper](https://arxiv.org/abs/2002.10698) [code](https://github.com/thaolmk54/hcrn-videoqa) |
| 6   |  SSML | 35.1 | [paper](https://arxiv.org/abs/2003.03186) |
| 7   |  HGA | 34.7 | [paper](https://ojs.aaai.org/index.php/AAAI/article/view/6767) [code](https://github.com/Jumpin2/HGA) |
| 8   |  HME | 33.7 | [paper](https://arxiv.org/pdf/1904.04357.pdf), [code](https://github.com/fanchenyou/HME-VideoQA) |
| 9   |  AMU | 32.0 | [paper](http://staff.ustc.edu.cn/~hexn/papers/mm17-videoQA.pdf), [code](https://github.com/xudejing/video-question-answering) |
| 10   |  ST-VQA | 31.3 | [paper](https://arxiv.org/pdf/1704.04497.pdf), [code](https://github.com/YunseokJANG/tgif-qa) |


## Auto-Downloading
```
cd lavis/datasets/download_scripts && python download_msvd.py
```

## References
Chen, David, and William B. Dolan. "Collecting highly parallel data for paraphrase evaluation." In Proceedings of the 49th annual meeting of the association for computational linguistics: human language technologies, pp. 190-200. 2011.

Xu, Dejing, Zhou Zhao, Jun Xiao, Fei Wu, Hanwang Zhang, Xiangnan He, and Yueting Zhuang. "Video question answering via gradually refined attention over appearance and motion." In Proceedings of the 25th ACM international conference on Multimedia, pp. 1645-1653. 2017.
