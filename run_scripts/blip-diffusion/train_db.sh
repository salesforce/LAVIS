SUBJECT_TEXT="dog"
IMAGE_STORAGE="/export/home/workspace/LAVIS-Diffusion/LAVIS/projects/blip-diffusion/images/dreambooth/dog"
MAX_ITERS=40
ITERS_PER_INNER_EPOCH=40 # number of iterations before saving a checkpoint 
BATCH_SIZE=3
LR=5e-6
WEIGHT_DECAY=0.01
OUTPUT_DIR="output/debug/BLIP-diffusion/finetune/dog"

python -m torch.distributed.run \
--nproc_per_node=1 train.py \
--cfg-path lavis/projects/blip_diffusion/finetune-db-template.yaml \
--options datasets.blip_diffusion_finetune.build_info.subject_text=$SUBJECT_TEXT \
          datasets.blip_diffusion_finetune.build_info.images.storage=$IMAGE_STORAGE \
          run.max_iters=$MAX_ITERS \
          run.iters_per_inner_epoch=$ITERS_PER_INNER_EPOCH \
          run.output_dir=$OUTPUT_DIR \
          run.init_lr=$LR \
          run.weight_decay=$WEIGHT_DECAY \
          run.batch_size_train=$BATCH_SIZE
