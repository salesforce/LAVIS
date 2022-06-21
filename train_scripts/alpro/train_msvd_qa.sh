cd ../..

python -m torch.distributed.run --nproc_per_node=16 train.py --cfg-path lavis/projects/alpro/exp_msvd_qa_ft.yaml
