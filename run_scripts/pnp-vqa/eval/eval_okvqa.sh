python -m torch.distributed.run --nproc_per_node=8 evaluate.py --cfg-path lavis/projects/pnp-vqa/eval/okvqa_eval.yaml
#python -m torch.distributed.run --nproc_per_node=1 evaluate.py --cfg-path lavis/projects/pnp-vqa/eval/okvqa_eval.yaml

