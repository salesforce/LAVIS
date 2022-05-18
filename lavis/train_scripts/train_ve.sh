cd ..

python -m torch.distributed.run --nproc_per_node=4 train.py --cfg-path projects/blip/exp_snli_ve_ft.yaml
