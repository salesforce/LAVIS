# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node=2 train.py --cfg-path lavis/projects/instructblip/train/pretrain.yaml
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=1 train.py --cfg-path lavis/projects/instructblip/train/pretrain.yaml
