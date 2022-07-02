python -m torch.distributed.run --nproc_per_node=8 --master_port 2345 evaluate.py --cfg-path lavis/projects/albef/eval/vqa_val.yaml
