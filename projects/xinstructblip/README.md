# X-InstructBLIP: A Framework for aligning X-Modal instruction-aware representations to LLMs and Emergent Cross-modal Reasoning

[Artemis Panagopoulou](https://artemisp.github.io), [Le Xue*](https://www.linkedin.com/in/le-tycho-xue-5abbb9157/), [Ning Yu*](https://ningyu1991.github.io/), [Junnan Li](https://sites.google.com/site/junnanlics), [Dongxu Li](https://sites.google.com/view/dongxu-li/home), [Shafiq Joty](https://scholar.google.com/citations?user=hR249csAAAAJ&hl=en&oi=ao), [Ran Xu](https://scholar.google.com/citations?user=sgBB2sUAAAAJ&hl=en), [Silvio Savarese](https://scholar.google.com/citations?user=ImpbxLsAAAAJ&hl=en), [Caiming Xiong](https://scholar.google.com/citations?user=vaSdahkAAAAJ&hl=en), and [Juan Carlos Niebles](https://scholar.google.com/citations?user=hqNhUCYAAAAJ&hl=en)

[![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg)](https://arxiv.org/pdf/2311.18799.pdf)  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/salesforce/LAVIS/blob/main/projects/xinstructblip/demo/run_demo.ipynb)
*equal mentorship contribution. 

## Overview

X-InstructBLIP a simple yet effective multimodal framework built on top of a frozen LLM, capable of seamlessly integrating and managing an ad-hoc number of modalities. Despite each modality projection being trained individually, X-InstructBLIP demonstrates joint reasoning abilities on par with models specifically trained on combined-modality datasets, such as video-audio.
![Architecture Overview](assets/architecture.png)

---
## Installation

### LAVIS Repository
```
git clone https://github.com/salesforce/LAVIS.git
cd LAVIS-XInstructBLIP
pip install -e .
```

### Evaluation Dependencies 
For the evaluation code, make sure that `java` is installed in the system by running
```
apt-get update
apt-get install default-jre
```

### KNN Cuda
Install [`KNN_Cuda`](https://github.com/unlimblue/KNN_CUDA) which is required for the 3D encoder. 
```
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```
Make sure that `ninja` is installed: 
```
wget -P /usr/bin https://github.com/unlimblue/KNN_CUDA/raw/master/ninja
```

### Install NLTK-CMUDICT
```
>>> import nltk
>>> nltk.download('cmudict')
```

## Pretrained Models
### Language Model Weights
First download the Vicuna v1.1 weights following the instructions [here](https://github.com/lm-sys/FastChat). Update the parameter `llm_model` in `configs/models/blip2/blip2_xinstruct_vicuna7b.yaml` and `configs/models/blip2/blip2_xinstruct_vicuna13b.yaml` and in the demo configs under `projects/xinstructblip/demo/configs` to the path of the downloaded model folder. 

### X-InstructBLIP Weights
Weights of the model are released [here (7b)](https://github.com/salesforce/LAVIS/blob/main/lavis/configs/models/blip2/blip2_xinstruct_vicuna7b.yaml) and [here (13b)](https://github.com/salesforce/LAVIS/blob/main/lavis/configs/models/blip2/blip2_xinstruct_vicuna13b.yaml) . When loading the model using the LAVIS codebase they should be automatically downloaded. 
```
from lavis.models import load_model
model = load_model("blip2_vicuna_xinstruct", "vicuna7b")
```


### BEATs pretrained model
Download the pretrained weights of the BEATs encoder [here](https://valle.blob.core.windows.net/share/BEATs/BEATs_iter3_plus_AS2M.pt?sv=2020-08-04&st=2023-03-01T07%3A51%3A05Z&se=2033-03-02T07%3A51%3A00Z&sr=c&sp=rl&sig=QJXmSJG9DbMKf48UDIU1MfzIro8HQOf3sqlNXiflY1I%3D) and update the key   
```
audio_encoder_kwargs :  {"checkpoint_path": "/path/to/BEATs_iter3_plus_AS2M.pt"}
```
in the X-InstructBLIP (7b) and (13b) configs found in `configs/models/blip2/blip2_xinstruct_vicuna7b.yaml` and `configs/models/blip2/blip2_xinstruct_vicuna13b.yaml` respectively.


## Config

X-InstructBLIP can be modified in various ways based on the config. Here is a documentation of each input field.

### General Settings
- **arch**: Architecture of the model, `blip2_vicuna_xinstruct`.
- **model_type**: Specific model variant, `vicuna7b`.
- **load_pretrained**: Boolean to load a pre-trained model (`True`/`False`).

### Pretrained Model Paths
- **pretrained**: Path to the pretrained model.
- **load_finetuned**: Load a finetuned model version (`True`/`False`).
- **finetuned**: Path to the finetuned model.

### Model Components
- **image_model**, **pc_model**, **video_model**, **audio_model**: Models for image, point cloud, video, and audio encoders.

### Pretrained Component Paths
- **pretrained_image_qformer**, **pretrained_pc_qformer**, **pretrained_video_qformer**: Paths to pretrained models for image, point cloud, and video qformers.

### Load Attention and Projection Settings
- Parameters for loading attention mechanisms and projections in image, PC, video, and audio components.
- **projection_only_{modality}**: use projection instead of q-former for specified modality
- **load_attention_{modality}_qformer**: load pretrained q-former cross-attention to text for particular modality
- **load_ln_type_{modality}**: key specification of encoder layer norm to load from state_dict. Will be of the form `{load_ln_type}_ln`.
- **load_qformer_type_{modality}**: key specification of q-former to load from state_dict. Will be of the form `{load_qformer_type}_qformer`.
- **load_projection_{modality}**: boolean whether to load pretrained LLM projection for modality.

### Encoder Settings
- **image_encoder_kwargs**, **pc_encoder_kwargs**, **video_encoder_kwargs**, **audio_encoder_kwargs**: Encoder arguments for various modalities.

### Precision and Freeze Settings
- **image_precision**, **pc_precision**, **video_precision**, **audio_precision**: Precision (e.g., "fp16") for different modalities.
- **freeze_image**, **freeze_pc**, **freeze_video**, **freeze_audio**: Freeze respective model components.

### Query and Text Settings
- **num_query_token**: Number of query tokens.
- **llm_model**: Path to the language model.
- **prompt**: Query prompt format.
- **max_txt_len**, **max_output_txt_len**: Max lengths for input and output text.
- **apply_lemmatizer**: Use lemmatizer on text.

### Text Input
- **qformer_text_input**, **llm_text_input**: Whether to use text input for q-former and language model.

### Modality and Cues
- **modalities**: List of used modalities, e.g., ["audio"].
- **use_cues**: Use cues in the model.

### Shared Q-Former Settings
- Shared qformer related parameters, including paths, load settings, and dimensions.

### Additional Settings
- **prefix**, **postfix**: Texts for model input.
- **predict_with_gen**: Model prediction includes generation.
- **proj_dim**: Projection hidden dimension, defaults to 1.
- **special_qformer_input_prompt**: different q-former input prompt than the llm model.



## Demo
Gradio based demo is available:
```
cd projects/xinstructblip/demo
python run_demo.py --model_type vicuna7b
```
The model configs can be adapted, and are found in `projects/xinstructblip/demo/configs`

## Result Replication
A collection of configs for training and evaluation of all the numbers reported in the paper are available at:
`lavis/projects/xinstructblip/train` and `lavis/projects/xinstructblip/eval`. For internal reproducibility the data paths are absolute paths in the servers. Replace those with the corresponding data links and storage paths for your data as downloaded in section <a name=Data>Data</a>.

Files can be run as follows:
```
python -m torch.distributed.run --nproc_per_node=8 train.py --cfg-path path/to/train/or/eval/config
```

for example to train the 3D Q-Former one can run 
```
python -m torch.distributed.run --nproc_per_node=8 train.py --cfg-path lavis/projects/xinstruct_blip/train/vicuna7b/pc_training.yaml 
```
It is recommended to use 40GB GPU RAM. If this is not available, results can be partially replicated by loading the Vicuna LLM model in 8bit instead of 16bit. More details [here](https://huggingface.co/docs/transformers/main_classes/quantization).


## Data
X-InstructBLIP is trained and evaluated in a collection of public and generated datasets as shown in the figure below. 
<img src="assets/data.png" alt="image" style="width:600px;height:auto;">


The data configuration files in `LAVIS-xgen_mm/configs/datasets` will automatically download the data annotation files when loading the datasets. The (image, audio, video, 3D) data can be downloaded from the following links. Once the datasets are downloaded, update the `storage` field in corresponding config file with the path in which the data is located. 
### Image
#### Train
* [Conceptual Captions 12m](https://github.com/google-research-datasets/conceptual-12m)
* [SBU Captions](https://huggingface.co/datasets/sbu_captions)
* [Visual Genome](https://homes.cs.washington.edu/~ranjay/visualgenome/index.html)
* [COCO](https://cocodataset.org/)
* CapFilt combination of CC3M, CC12M, and SBU
* [LLaVa150k](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K)
* VGQA: see Visual Genome
* [VQAv2](https://visualqa.org/)
* [OKVQA](https://okvqa.allenai.org/)
* [AOKVQA](https://allenai.org/project/a-okvqa/home)
* [OCRVQA](https://ocr-vqa.github.io/)

#### Eval
* [GQA](https://cs.stanford.edu/people/dorarad/gqa/about.html)
* [VizWiz](https://vizwiz.org/tasks-and-datasets/vqa/)
* [Flickr30k](https://github.com/BryanPlummer/flickr30k_entities)
* [NoCaps](https://nocaps.org/)

### Audio
#### Train
* [AudioSet](http://research.google.com/audioset/)
* [AudioCaps](https://audiocaps.github.io/)
* [WavCaps](https://github.com/XinhaoMei/WavCaps)

#### Eval 
* [Clotho](https://zenodo.org/records/3490684)
* [ClothoAQA](https://zenodo.org/records/6473207)
* [ESC50](https://github.com/karolpiczak/ESC-50)
* AudioCaps for DisCRn

### Video
#### Train
* [WebVid2m](https://github.com/m-bain/webvid)
* [MSRVTTT](https://cove.thecvf.com/datasets/839)
#### Eval
* [MSVD](http://www.cs.utexas.edu/users/ml/clamp/videoDescription/)
* [MusicAVQA](https://gewu-lab.github.io/MUSIC-AVQA/)
* [VATEX](https://eric-xw.github.io/vatex-website/about.html)


### 3D

* [Cap3D](https://huggingface.co/datasets/tiange/Cap3D)


## Data Augmentation

We release the scripts to generate data both for the training data augmentation and generating the DisCRn dataset.

### QA Data Augmentation

The files `projects/xinstructblip/data_aug/3d_qa_data_generation.py` and `projects/xinstructblip/data_aug/audio_qa_data_generation.py` are used to generate the 3D and Audio QA data from Cap3D and Audiocaps respectively. 

#### 3DQA
Download the captions for the objaverse data [here](https://huggingface.co/datasets/tiange/Cap3D/blob/main/Cap3D_automated_Objaverse_no3Dword.csv). Then the script takes 4 arguments:
* `shard`: 1 through 4, for speed the data is processed in 4 batches and can be spread across machines or run consecutively in the same mahine
* `mode`: `color_removal` removes color concepts form captions since the encoder does not encode color, `qa_gen` generates question answer pairs, and `rtc` performs the roundtrip consistency check, `all` runs all of the process in this order. 
* `split`: `train` or `val`. Assumes a separation of the data which can be found in these [`train`](https://storage.cloud.google.com/sfr-xinstructblip-data-research/data/objaverse/cap3d_cap_final_train.csv) and [`dev`](https://storage.cloud.google.com/sfr-xinstructblip-data-research/data/objaverse/cap3d_cap_final_val.csv) files. If the split is not added to the csv then it can be set arbitrarily and all the data in the file will be processed. 
* `original_data_file`: the path to the captions for objaverse downloaded above. 

#### AudioQA
Download the Audiocaps captions from [here](https://github.com/cdjkim/audiocaps/tree/master/dataset). The script takes similar arguments as above:
* `shard`: 1 through 4, for speed the data is processed in 4 batches and can be spread across machines or run consecutively in the same mahine
* `mode`: `qa_gen` generates question answer pairs, and `rtc` performs the roundtrip consistency check, `all` runs all of the process in this order. 
* `split`: `train`, `val`, `test`. 
* `original_data_file`: the path to the captions for Audiocaps downloaded above for the relevant split. 

### DisCRn
The dataset is found here: [Audio-Video](https://storage.cloud.google.com/sfr-xinstructblip-data-research/data/discrn/audiocaps.json) and [Image-3D](https://storage.cloud.google.com/sfr-xinstructblip-data-research/data/discrn/objaverse.json). 
The files `projects/xinstructblip/discrn/data_generation/objaverse_img_3d.py` are `projects/xinstructblip/discrn/data_generation/audiocaps_video_audio.py` generate the image-3d and audio-video cross-modal reasoning pairs for the DisCRn task.
#### Image-3D
The arguments are as above, with the same 3D caption data
* `shard`
* `original_data_file`
* `split`
* `mode`: in this case it can take the values `property` to generate properties, `get_pairs` to identify new pairs,`instruction_gen` to generate instruction answer pairs and `rtc` to perform round trip consistency. `all` will run them all in order. 
* `rnd`: adds identifier in output files in the case of multiple generations. 


#### Audio Video
The arguments are as above, with the same audio caption data. Note that you should update `VIDEO_PATH` in the script with the equivalent downloaded videos from Audiocaps. An easy way to do so is using the `youtube-id` associated with each of them and a package such as [`youtube-dl`](`https://github.com/ytdl-org/youtube-dl`)
* `shard`
* `original_data_file`
* `split`
* `mode`: in this case it can take the values `property` to generate properties, `get_pairs` to identify new pairs,`instruction_gen` to generate instruction answer pairs and `rtc` to perform round trip consistency. `all` will run them all in order. 
* `rnd`: adds identifier in output files in the case of multiple generations. 


## Cite

```
@misc{panagopoulou2023xinstructblip,
    title={X-InstructBLIP: A Framework for aligning X-Modal instruction-aware 
        representations to LLMs and Emergent Cross-modal Reasoning}, 
    author={Artemis Panagopoulou and Le Xue and Ning Yu and Junnan Li and Dongxu Li and 
        Shafiq Joty and Ran Xu and Silvio Savarese and Caiming Xiong and Juan Carlos Niebles},
    year={2023},
    eprint={2311.18799},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
    }
```
