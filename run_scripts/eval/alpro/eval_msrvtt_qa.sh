python -m torch.distributed.run --nproc_per_node=16 evaluate.py --cfg-path lavis/projects/alpro/exp_msrvtt_qa_eval.yaml --options run.output_dir="output/ALPRO/msrvtt_qa"
