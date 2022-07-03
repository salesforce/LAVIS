# python -m torch.distributed.run --nproc_per_node=8 evaluate.py --cfg-path lavis/projects/blip/exp_coco_ret_eval.yaml
python -m torch.distributed.run --nproc_per_node=16 evaluate.py --cfg-path lavis/projects/blip/eval/ret_coco_eval.yaml
