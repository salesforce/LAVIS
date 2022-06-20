cd ../..

python -m torch.distributed.run --nproc_per_node=16 train.py --cfg-path lavis/projects/alpro/exp_msrvtt_qa_ft.yaml
