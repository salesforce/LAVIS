python -m torch.distributed.run --nproc_per_node=16 evaluate.py --cfg-path lavis/projects/alpro/exp_didemo_ret_eval.yaml --options run.output_dir="output/ALPRO/didemo"
