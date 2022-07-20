python -m torch.distributed.run --nproc_per_node=8 train.py --cfg-path lavis/projects/albef/train/snli_ve_ft.yaml
# CUDA_VISIBLE_DEVICES=8,9,10,11,12,13,14,15 python -m torch.distributed.run --nproc_per_node=8 --master_port 47770 train.py --cfg-path lavis/projects/albef/train/snli_ve_ft.yaml
