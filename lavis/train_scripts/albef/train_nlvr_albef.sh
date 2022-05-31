cd ../..

python -m torch.distributed.run --nproc_per_node=8 train.py --cfg-path projects/albef/exp_nlvr_ft.yaml
