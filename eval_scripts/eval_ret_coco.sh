cd ..

python -m torch.distributed.run --nproc_per_node=16 evaluate.py --cfg-path lavis/projects/blip/exp_coco_ret_eval.yaml
