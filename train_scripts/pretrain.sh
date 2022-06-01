cd ..

python -m torch.distributed.run --nproc_per_node=16 train.py --cfg-path lavis/projects/blip/exp_pretrain.yaml
