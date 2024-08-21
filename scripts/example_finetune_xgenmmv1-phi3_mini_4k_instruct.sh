#!/bin/bash

datamix=$1

exp_name="finetune-xgenmmv1-phi3_4k_instruct-${datamix}"


data_path="data_configs/${datamix}.yaml"

if [[ ! -e $exp_name ]]; then
    mkdir $exp_name
fi

pretrained_ckpt="/path/to/the/base/model.pt"

python -m torch.distributed.run --nproc_per_node=8 --nnodes=1 --master_port 9650 open_flamingo/train/instruction_finetune.py \
    --lm_path microsoft/Phi-3-mini-4k-instruct \
    --tokenizer_path microsoft/Phi-3-mini-4k-instruct \
    --conv_template_name phi_3 \
    --vision_encoder_path google/siglip-so400m-patch14-384 \
    --vision_encoder_pretrained google \
    --model_family 'xgenmm_v1' \
    --num_vision_tokens 128 \
    --pretrained ${pretrained_ckpt} \
    --data_path ${data_path} \
    --data_sampler_group_by_length \
    --image_aspect_ratio anyres --anyres_patch_sampling \
    --batch_size 8 \
    --fsdp \
    --no_save_optim_state \
    --gradient_checkpointing \
    --fsdp_sharding_strategy hybrid \
    --workers 4 \
    --num_epochs 1 \
    --warmup_steps  2000 \
    --learning_rate 2e-5 \
    --weight_decay 0.0 \
    --lr_scheduler cosine \
    --precision amp_bf16 \
    --report_to_wandb \
    --wandb_project "blip3-xgenmm-finetune" \
    --run_name ${exp_name} 2>&1 | tee ${exp_name}/terminal_output.log;
