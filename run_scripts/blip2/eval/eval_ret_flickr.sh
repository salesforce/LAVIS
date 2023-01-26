# python -m torch.distributed.run --nproc_per_node=16 evaluate.py --cfg-path lavis/projects/blip2/eval/ret_flickr_eval.yaml
python -m torch.distributed.run --nproc_per_node=8 evaluate.py --cfg-path lavis/projects/blip2/eval/ret_flickr_eval.yaml
