""" Main training script """
import argparse
from datetime import datetime
import os
from omegaconf import OmegaConf
import torch
import wandb
import functools
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy

from open_flamingo import create_model_and_transforms, SUPPORTED_MODEL_FAMILIES
from open_flamingo.train.distributed import (
    init_distributed_device,
    world_info_from_env,
    get_fsdp_config,
    get_fsdp_checkpoint_config,
)
from open_flamingo.train.sft_data_utils import make_supervised_data_module
from open_flamingo.train.train_utils import (
    finetune_one_epoch,
    random_seed,
    find_most_recent_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from open_flamingo.train.losses import (
    SUPPORTED_LOSSES,
    get_loss_fn,
)
from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

def parse_tuple_list(input_string):
    try:
        tuples = input_string.strip().strip('()').split('),(')
        # Convert each item in the list to a tuple
        tuple_list = [tuple(map(int, item.split(','))) for item in tuples]
        return tuple_list
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid tuple list format: {input_string}. Error: {e}")


def main():
    parser = argparse.ArgumentParser()
    # model configuration args
    parser.add_argument(
        "--model_family", default="kosmos-instruct", type=str, choices=SUPPORTED_MODEL_FAMILIES
    )
    parser.add_argument("--vision_encoder_path", default="ViT-SO400M-14-SigLIP-384", type=str)
    parser.add_argument("--vision_encoder_pretrained", default="webli", type=str)
    parser.add_argument("--lm_path", default="facebook/opt-1.3b", type=str)
    parser.add_argument(
        "--tokenizer_path",
        default="facebook/opt-30b",
        type=str,
        help="path to tokenizer",
    )
    parser.add_argument(
        "--cross_attn_every_n_layers",
        type=int,
        default=1,
        help="how often to add a cross-attention layer after each transformer layer",
    )
    parser.add_argument(
        "--num_vision_tokens",
        type=int, default=64, help="number of query tokens used for resampling vision features.",
    )
    parser.add_argument("--pretrained", type=str, default=None, help="pretrained weights for fine-tuning.")
    parser.add_argument("--pretrained_vision_tokenizer", type=str, default=None, help="pretrained vl connector for fine-tuning.")
    
    # training args
    parser.add_argument(
        "--loss", type=str, choices=SUPPORTED_LOSSES, default="supervised_finetune"
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="openflamingo3B",
        help="used to name saving directory and wandb run",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        help="path to checkpoint to resume from, this should contain model, optimizer, and lr_scheduler states. if there exists a checkpoint in the dir named run_name, we will resume from that checkpoint by default.",
        default=None,
    )
    parser.add_argument(
        "--delete_previous_checkpoint",
        action="store_true",
        help="delete previous checkpoint when saving new checkpoint",
    )
    parser.add_argument(
        "--no_save_optim_state",
        action="store_true",
        help="do not save optimizer states when saving checkpoints",
    )
    
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument(
        "--lr_scheduler",
        default="constant",
        type=str,
        help="constant, linear, or cosine",
    )
    parser.add_argument("--warmup_steps", default=5000, type=int)
    parser.add_argument("--weight_decay", default=0.1, type=float)
    parser.add_argument(
        "--precision",
        choices=["amp_bf16", "amp_bfloat16", "bf16", "fp16", "fp32"],
        default="fp32",
        help="Floating point precision.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="whether to train with gradient/activation checkpointing",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help="we define an 'epoch' as a fixed number of examples specified by train_num_samples, not a pass through the entire dataset",
    )
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--logging_steps", type=int, default=100, help="log loss every n steps"
    )
    parser.add_argument(
        "--checkpoint_steps", type=int, default=5000, help="log loss every n steps"
    )

    # data args
    # TODO: load a data args yaml file
    parser.add_argument(
        "--data_path", 
        default="/export/home/LLaVA/playground/data/llava_v1_5_mix665k_ocr_tagged_vqa_placeholder.json", 
        type=str
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--data_sampler_group_by_length", default=False, action="store_true")

    # Legacy Llava data args
    parser.add_argument("--is_multimodal", type=bool, default=True)
    parser.add_argument("--mm_use_im_start_end", default=False, action="store_true")
    parser.add_argument("--conv_template_name", type=str, default=None)

    # Any resolution
    parser.add_argument("--image_aspect_ratio", type=str, default='pad')
    parser.add_argument(
        "--anyres_patch_sampling",
        default=False,
        action="store_true",
    )
    parser.add_argument('--anyres_grids', 
                        type=parse_tuple_list, 
                        default="(1,2),(2,1),(2,2),(3,1),(1,3)",
                        help="List of tuples in the format (1,2),(3,4),...")
    
    
    # distributed training args
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--horovod",
        default=False,
        action="store_true",
        help="Use horovod for distributed training.",
    )
    parser.add_argument(
        "--no-set-device-rank",
        default=False,
        action="store_true",
        help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc).",
    )
    parser.add_argument(
        '--local-rank',
        default=0,
        type=int,
        help='Local rank for distributed training'
    )

    # fsdp args
    parser.add_argument(
        "--fsdp",
        default=False,
        action="store_true",
        help="Use FullyShardedDataParallel for distributed training. Not supported for some models, e.g. OPT.",
    )
    parser.add_argument(
        "--fsdp_sharding_strategy", default="full", type=str, choices=["full", "hybrid", "shard_grad_op", "hybrid_shard_grad_op", "no_shard"]
    )

    # wandb args
    parser.add_argument("--report_to_wandb", default=False, action="store_true")
    parser.add_argument(
        "--wandb_project",
        type=str,
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
    )
    parser.add_argument(
        "--save_checkpoints_to_wandb",
        default=False,
        action="store_true",
        help="save checkpoints to wandb",
    )
    parser.add_argument(
        "--dryrun",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        '--use_flash_attention_2',
        default=False,  action='store_true',
        help='Use Flash Attention 2.0 for language model.'
    )
    parser.add_argument(
        '--unfreeze_vision_encoder',
        default=False,  action='store_true',
        help='Unfreeze vision encoder during training.'
    )
    parser.add_argument(
        '--vision_encoder_precision',
        default='fp32',
        choices=["bf16", "fp32"],
        help='Precision of the vision encoder during training.'
    )
    parser.add_argument(
        '--cpu_offload_gradients',
        default=False,  action='store_true',
        help='This specifies whether to offload parameters to CPU when not involved in computation. If True, then this offloads gradients to CPU as well, meaning that the optimizer step runs on CPU.'
    )


    args = parser.parse_args()


    if args.save_checkpoints_to_wandb and not args.report_to_wandb:
        raise ValueError("save_checkpoints_to_wandb requires report_to_wandb")

    if args.fsdp:
        assert (
            torch.__version__ > "2.0.1"
        ), "FSDP requires torch > 2.0.1"

    # Set up distributed training
    args.local_rank, args.rank, args.world_size = world_info_from_env()
    if args.rank == 0:
        print(f"Initializing distributed training with {args.world_size} GPUs.")
    if args.offline:
        os.environ["WANDB_MODE"] = "offline"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    device_id = init_distributed_device(args)

    random_seed(args.seed)

    # Initialize model
    if args.model_family == "flamingo":
        additional_kwargs={"cross_attn_every_n_layers": args.cross_attn_every_n_layers}
    elif args.model_family in ['xgenmm_v1']:
        additional_kwargs = {
            "image_aspect_ratio": args.image_aspect_ratio,
            "num_vision_tokens": args.num_vision_tokens,
            "anyres_patch_sampling": args.anyres_patch_sampling,
        }
    else:
        additional_kwargs = {}
        
    model, image_processor, tokenizer = create_model_and_transforms(
        args.vision_encoder_path,
        args.vision_encoder_pretrained,
        args.lm_path,
        args.tokenizer_path if args.tokenizer_path else args.lm_path,
        model_family=args.model_family,
        pretrained_vision_tokenizer=args.pretrained_vision_tokenizer,
        use_local_files=args.offline,
        gradient_checkpointing=args.gradient_checkpointing,
        verbose=(args.rank == 0),
        **additional_kwargs,
    )
    random_seed(args.seed, args.rank)

    # Initialize wandb logging
    now = datetime.now().strftime("%Y%m%d%H%M")[:-1]
    if args.rank == 0 and args.report_to_wandb:
        wandb.init(
            project=args.wandb_project,
            name=f"{args.run_name}-{now}",
            config=vars(args),
        )

    # Load model checkpoint (on CPU)
    if args.fsdp:
        args.fsdp_checkpoint_config = get_fsdp_checkpoint_config(args)

    # if args do not specify a checkpoint to resume from, resume from most recent checkpoint
    resume_from_step = 0
    if os.path.exists(f"{args.run_name}") and args.resume_from_checkpoint is None:
        args.resume_from_checkpoint = find_most_recent_checkpoint(args)

    if (
        args.resume_from_checkpoint is not None
    ): 
        resume_from_epoch, resume_from_step, checkpoint = load_checkpoint(args, model)
        print(f"Resume training from epoch {resume_from_epoch}, step {resume_from_step}...")
    else:
        resume_from_epoch = 0
        resume_from_step = 0
    
    # Load pretrained weights.
    if args.resume_from_checkpoint is None and not args.dryrun:
        if args.pretrained_vision_tokenizer is None:
            assert os.path.exists(args.pretrained), "Must fine-tune from a pretrained weight."
        if args.pretrained is not None:
            _, _, checkpoint = load_checkpoint(args, model, pretrained=True)
            print("Finished loading checkpoint...")

    # Initialize gradient checkpointing
    if args.gradient_checkpointing:
        model.init_gradient_checkpointing()

    # Initialize FSDP / DDP, and ensure the model is on GPU
    if args.fsdp:
        auto_wrap_policy = functools.partial(
            lambda_auto_wrap_policy, lambda_fn=model.get_fsdp_lambda_fn()
        )
        wrapper_kwargs = get_fsdp_config(args, device_id)
        distributed_model = FSDP(
            model, auto_wrap_policy=auto_wrap_policy, **wrapper_kwargs
        )
        print("Finished FSDP wrapping...")
    else:
        model = model.to(device_id)
        distributed_model = DDP(model, device_ids=[device_id])

    # Initialize optimizer
    params_with_wd, params_without_wd = model.group_params_by_weight_decay()
    optimizer = torch.optim.AdamW(
        [
            {"params": params_with_wd, "weight_decay": args.weight_decay},
            {"params": params_without_wd, "weight_decay": 0.0},
        ],
        lr=args.learning_rate,
    )

    # load optimizer checkpoint
    if args.resume_from_checkpoint is not None:
        optim_state_dict = checkpoint["optimizer_state_dict"]
        if args.fsdp:
            # FSDP.set_state_dict_type(
            #     distributed_model,
            #     **args.fsdp_checkpoint_config,
            # )
            optim_state_dict = FSDP.optim_state_dict_to_load(
                model=distributed_model, optim=optimizer, optim_state_dict=optim_state_dict
            )
        optimizer.load_state_dict(optim_state_dict)

    # Initialize datasets
    if args.data_path.split('.')[-1] == 'yaml':
        # Loading a mixture of datasets with sampling ratios.
        data_config = OmegaConf.load(args.data_path)
        if args.rank == 0:
            print("================== Data mixture config ===================")
            print(data_config)
            print("==========================================================")
        args.data_path = dict(data_config.data_path)
    train_dataset, total_num_samples = make_supervised_data_module(tokenizer=tokenizer, 
                                                                   image_processor=image_processor, 
                                                                   data_args=args)
    # Update anyres grid.
    args.anyres_grids = train_dataset.dataloader.dataset.anyres_grids
    model.anyres_grids = args.anyres_grids

    # TODO: Summarize training data stats (dataset, portion, etc.)
    total_training_steps = (
        total_num_samples
        // (args.batch_size * args.gradient_accumulation_steps * args.world_size)
    ) * args.num_epochs

    if args.rank == 0:
        print(f"Total training steps: {total_training_steps}")

    # Initialize lr scheduler
    if args.lr_scheduler == "linear":
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=total_training_steps,
        )
    elif args.lr_scheduler == "cosine":
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=total_training_steps,
        )
    else:
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps
        )

    # load lr scheduler checkpoint
    if args.resume_from_checkpoint is not None:
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])

    # Initialize the loss fn
    loss_fn = get_loss_fn(args.loss)

    # check wrapping
    if args.rank == 0:
        print(distributed_model)

    # Start training!
    print(f"Start running training on rank {args.rank}.")
    for epoch in range(resume_from_epoch, args.num_epochs):
        train_dataset.set_epoch(epoch)
        finetune_one_epoch(
            args=args,
            resume_from_step=resume_from_step,
            model=distributed_model,
            epoch=epoch,
            dataset=train_dataset,
            compute_loss_fn=loss_fn,
            tokenizer=tokenizer,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            device_id=device_id,
            wandb=wandb,
        )

        save_checkpoint(distributed_model, optimizer, lr_scheduler, epoch, args)


if __name__ == "__main__":
    main()