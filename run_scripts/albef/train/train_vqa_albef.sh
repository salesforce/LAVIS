# python -m torch.distributed.run --nproc_per_node=8 train.py --cfg-path lavis/projects/albef/train/vqa_ft.yaml
python -m torch.distributed.run --nproc_per_node=8 train.py --cfg-path lavis/projects_release/albef/train/vqa_ft.yaml
