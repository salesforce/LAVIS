cd ..

python -m torch.distributed.run --nproc_per_node=16 train.py --cfg-path projects/blip/exp_vqa_ft.yaml

# python train.py --cfg-run projects/blip/run.yaml \
# 	            --cfg-data projects/blip/dataset_cap.yaml \
#  		        --cfg-model projects/blip/model.yaml
				#  --options model.model_type="large"

# python -m torch.distributed.run --nproc_per_node=2 train.py \
# 				--cfg-run projects/blip/run.yaml \
# 	            --cfg-data projects/blip/dataset_cap.yaml \
#  		        --cfg-model projects/blip/model.yaml
