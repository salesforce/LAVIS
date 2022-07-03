# python -m torch.distributed.run --nproc_per_node=8 evaluate.py --cfg-path lavis/projects/albef/eval/coco_retrieval_eval.yaml
python -m torch.distributed.run --nproc_per_node=8 evaluate.py --cfg-path lavis/projects/albef/eval/ret_coco_eval.yaml
