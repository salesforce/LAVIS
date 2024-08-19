# OpenFlamingo Training
We provide efficient data loading and distributed training code.
To train with OpenFlamingo, please ensure your environment matches that of `environment.yml`.

Table of contents:

* [Data](#data)
* [Example commands](#example-training-command)
* [Distributed training](#distributed-training)

## Data
Our codebase uses [WebDataset](https://github.com/webdataset/webdataset) to efficiently load `.tar` files containing image and text sequences. We recommend resampling shards with replacement during training using the `--dataset_resampled` flag. 

Supported pretraining datasets
* LAION-2B
* Multimodal C4 (MMC4)
* ChatGPT-generated sequences from OpenFlamingo [technical report](https://arxiv.org/abs/2308.01390)

We plan to add additional datasets in the future, and we welcome contributions! If you'd like to add support for a pretraining dataset, please open a PR.

### LAION-2B Dataset
[LAION-2B](https://arxiv.org/abs/2210.08402) contains 2B web-scraped (image, text) pairs. 
We use [img2dataset](https://github.com/rom1504/img2dataset) to download this dataset into tar files.

### Multimodal C4 Dataset
We train on the full version of [Multimodal C4 (MMC4)](https://github.com/allenai/mmc4), which includes 103M documents of web-scraped, interleaved image-text sequences. During training, we truncate sequences to 256 text tokens and six images per sequence.

Our codebase expects `.tar` files containing `.json` files, which include raw images encoded in base64.
We provide scripts to convert MMC4 to this format: 

1. Download the MMC4 shards into `.zip` files using [the MMC4-provided scripts](https://github.com/allenai/mmc4/tree/main/scripts) (e.g., `fewer_facesv2.sh`).
2. Download the MMC4 raw images into an image directory using [the MMC4-provided scripts](https://github.com/allenai/mmc4/tree/main/scripts) (e.g., `download_images.py`).
2. Run `scripts/convert_mmc4_to_wds.py` to convert the downloaded items into the expected tar files.

### ChatGPT-generated sequences
A subset of our models (listed below) were also trained on experimental ChatGPT-generated (image, text) sequences, where images are pulled from LAION. The shards containing these sequences can be found at [this CodaLab worksheet](https://worksheets.codalab.org/worksheets/0xdcd888ff7c754ae680c5e038f6ed1d9b). We are unable to distribute raw images in the released shards; images must be pre-downloaded from the urls in the json files and converted to base64 before using this data for training in our codebase.

Models trained with ChatGPT-generated sequences:

* OpenFlamingo-4B-vitl-rpj3b
* OpenFlamingo-4B-vitl-rpj3b-langinstruct

## Example training command
We provide sample Slurm training scripts in `scripts/`. You can also modify the following command:

```
torchrun --nnodes=1 --nproc_per_node=4 train.py \
  --lm_path anas-awadalla/mpt-1b-redpajama-200b \
  --tokenizer_path anas-awadalla/mpt-1b-redpajama-200b \
  --cross_attn_every_n_layers 1 \
  --dataset_resampled \
  --batch_size_mmc4 32 \
  --batch_size_laion 64 \
  --train_num_samples_mmc4 125000\
  --train_num_samples_laion 250000 \
  --loss_multiplier_laion 0.2 \
  --workers=4 \
  --run_name OpenFlamingo-3B-vitl-mpt1b \
  --num_epochs 480 \
  --warmup_steps  1875 \
  --mmc4_textsim_threshold 0.24 \
  --laion_shards "/path/to/shards/shard-{0000..0999}.tar" \
  --mmc4_shards "/path/to/shards/shard-{0000..0999}.tar" \
  --report_to_wandb
```
*Note: The MPT-1B [base](https://huggingface.co/mosaicml/mpt-1b-redpajama-200b)  and [instruct](https://huggingface.co/mosaicml/mpt-1b-redpajama-200b-dolly) modeling code does not accept the `labels` kwarg or compute cross-entropy loss directly within `forward()`, as expected by our codebase. We suggest using a modified version of the MPT-1B models found [here](https://huggingface.co/anas-awadalla/mpt-1b-redpajama-200b) and [here](https://huggingface.co/anas-awadalla/mpt-1b-redpajama-200b-dolly).*

## Distributed training
Our codebase supports distributed training using three frameworks:

* Pytorch's [DistributedDataParallel](https://pytorch.org/docs/stable/torch.nn.parallel.DistributedDataParallel.html). This is the default method used by `train.py`.
* Pytorch's [FullyShardedDataParallel](https://pytorch.org/docs/stable/fsdp.html) (FSDP). Use the `--fsdp` flag.

Note that you should use exactly one of these training methods.

`train/distributed.py` contains utilities to help with setting up distributed training using Slurm / `torchrun`. See example scripts in the `scripts` directory.

### FSDP notes
To use FSDP, make sure to use Pytorch (> 2.0.1). 

We support two sharding strategies for FSDP: full sharding (model sharing across all nodes and GPUs) or hybrid sharding (model sharding across GPUs within nodes, data parallel between nodes). The former saves GPU memory; the latter saves on communication costs.
