import time
from contextlib import suppress
import torch
from tqdm import tqdm
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import os
import shutil
import wandb
import glob
from data_utils import DataInfo
import random
import numpy as np
import torch.nn as nn


def train_one_epoch(
    args,
    model,
    epoch,
    datasets: [DataInfo],
    compute_loss_fn: callable,
    tokenizer,
    optimizer,
    lr_scheduler,
    device_id,
    wandb,
):
    """
    Helper function for running one epoch of training.
    Handles logging, calling forward, backward, gradient clipping, and optimizer step.
    Args:
        args (argparse.Namespace): arguments from command line
        model: DDP / FSDP wrapped model
        epoch (int): epoch number
        datasets (list): list of DataInfos, one for each dataset, to train on
        compute_loss_fn (callable): function that given the model and inputs, calls forward
            and returns a loss
        tokenizer: tokenizer for the language model
        optimizer: optimizer to step
        lr_scheduler: learning rate scheduler
        device_id (int): GPU device ID for this rank
        wandb: wandb object for logging
    """
    # calculate the number of steps in an epoch
    num_batches_per_epoch = datasets[0].dataloader.num_batches
    total_training_steps = num_batches_per_epoch * args.num_epochs

    # set up model, autocast, and dtypes
    model.train()
    autocast = get_autocast(args.precision)

    # set up logging
    step_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    # loop through the batches in this epoch
    for step_num, batches in tqdm(
        enumerate(zip(*[dataset.dataloader for dataset in datasets])),
        disable=args.rank != 0,
        total=total_training_steps,
        initial=(epoch * num_batches_per_epoch),
    ):
        data_time_m.update(time.time() - end)
        global_step = step_num + epoch * num_batches_per_epoch

        # call compute_loss_fn on each dataset; call backward before continuing
        losses_to_log = {}
        batch_metadata_to_log = {}
        for dataset_ix, (images, (input_ids, attention_mask)) in enumerate(batches):
            # unpack the batch and move to device
            images = images.to(device_id, non_blocking=True)
            input_ids = input_ids.to(device_id, non_blocking=True)
            attention_mask = attention_mask.to(device_id, non_blocking=True)

            # save some metadata for logging
            batch_metadata_to_log[
                f"{datasets[dataset_ix].name}_num_tokens"
            ] = attention_mask.sum().item()
            batch_metadata_to_log[f"{datasets[dataset_ix].name}_num_images"] = (
                (input_ids == unwrap_model(model).media_token_id).sum().item()
            )

            # forward pass
            dataset_loss = compute_loss_fn(
                model=model,
                tokenizer=tokenizer,
                images=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )[0]

        divided_loss_laion = loss_laion / args.gradient_accumulation_steps
        (divided_loss_laion * args.loss_multiplier_laion).backward()

        #### MMC4 FORWARD PASS ####
        images = batch_mmc4[0].to(device_id, dtype=cast_dtype, non_blocking=True)
        images = rearrange(images, "b (t f) c h w -> b t f c h w", f=1)
        input_ids = torch.stack([x[0] for x in batch_mmc4[1]]).squeeze(1)
        attention_mask = torch.stack([x[1] for x in batch_mmc4[1]]).squeeze(1)

        # set up labels; language model is expected to handle shifting
        labels = input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100
        labels[labels == tokenizer.eos_token] = -100
        for i in range(labels.shape[0]):
            # remove loss for any token before the first <image> token
            label_idx = 0
            while (
                label_idx < labels.shape[1] and labels[i][label_idx] != media_token_id
            ):
                labels[i][label_idx] = -100
                label_idx += 1

            # get index of all endofchunk tokens in the sequence
            endofchunk_idxs = torch.where(labels[i] == endofchunk_token_id)[0]
            for endofchunk_idx in endofchunk_idxs:
                token_idx = endofchunk_idx + 1
                while (
                    token_idx < labels.shape[1]
                    and labels[i][token_idx] != media_token_id
                ):
                    labels[i][token_idx] = -100
                    token_idx += 1

        labels[labels == media_token_id] = -100
        labels = labels.to(device_id)

        # gradient accumulation w/ fsdp cpu offloading requires a no_sync context manager
        with autocast():
            loss_mmc4 = model(
                vision_x=images,
                lang_x=input_ids.to(device_id),
                attention_mask=attention_mask.to(device_id),
                labels=labels,
            )[0]

            # if loss is nan, skip this batch
            # this hack of skipping the batch is not FSDP-compatible
            if torch.isnan(loss_mmc4):
                print("loss is nan, skipping this batch")
                print("input_ids: ", tokenizer.batch_decode(input_ids))
                print("labels: ", labels)
                print("images: ", images)
                optimizer.zero_grad(set_to_none=True)
                continue

        divided_loss_mmc4 = loss_mmc4 / args.gradient_accumulation_steps
        (divided_loss_mmc4 * args.loss_multiplier_mmc4).backward()

        if (not args.freeze_lm_embeddings) and (
            not args.fsdp or args.fsdp_use_orig_params
        ):
            # Mask gradients for input embeddings s.t. we only update the added tokens <image> and <|endofchunk|>
            if args.fsdp:
                embed_grad = model.lang_encoder.get_input_embeddings().weight.grad
            else:
                embed_grad = (
                    model.module.lang_encoder.get_input_embeddings().weight.grad
                )
            zero_mask = torch.zeros_like(embed_grad)
            zero_mask[media_token_id] = torch.ones_like(zero_mask[media_token_id])
            zero_mask[endofchunk_token_id] = torch.ones_like(
                zero_mask[endofchunk_token_id]
            )
            if args.fsdp:
                model.lang_encoder.get_input_embeddings().weight.grad = (
                    embed_grad * zero_mask
                )
            else:
                model.module.lang_encoder.get_input_embeddings().weight.grad = (
                    embed_grad * zero_mask
                )

        # clip gradient norm
        if args.fsdp:
            model.clip_grad_norm_(1.0, norm_type=2.0)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # step optimizer and log
        if (((step_num + 1) % args.gradient_accumulation_steps) == 0) or (
            step_num == num_batches_per_epoch - 1
        ):
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            # step time and reset end outside of rank 0
            step_time_m.update(time.time() - end)
            end = time.time()

            # rank 0 logging
            if args.rank == 0 and args.report_to_wandb:
                # calculate samples per second
                throughput_metrics = compute_throughput(
                    args,
                    datasets,
                    batch_metadata_to_log,
                    step_time_m,
                )
                wandb.log(
                    {
                        "global_step": global_step,
                        "lr": optimizer.param_groups[0]["lr"],
                        "data_time": data_time_m.avg,
                        "step_time": step_time_m.avg,
                        **throughput_metrics,
                        **losses_to_log,
                    },
                    commit=True,
                )
                step_time_m.reset()
                data_time_m.reset()

        # Log loss to console
        if ((step_num + 1) % args.logging_steps == 0) and args.rank == 0:
            print(
                f"Step {step_num+1}/{num_batches_per_epoch} of epoch {epoch+1}/{args.num_epochs} complete. Losses: "
                + "// ".join([f"{k}: {v:.3f}" for k, v in losses_to_log.items()])
            )

def finetune_one_epoch(
    args,
    resume_from_step,
    model,
    epoch,
    dataset: DataInfo,
    compute_loss_fn: callable,
    tokenizer,
    optimizer,
    lr_scheduler,
    device_id,
    wandb,
):
    """
    Helper function for running one epoch of training.
    Handles logging, calling forward, backward, gradient clipping, and optimizer step.
    Args:
        args (argparse.Namespace): arguments from command line
        model: DDP / FSDP wrapped model
        epoch (int): epoch number
        datasets (list): list of DataInfos, one for each dataset, to train on
        compute_loss_fn (callable): function that given the model and inputs, calls forward
            and returns a loss
        tokenizer: tokenizer for the language model
        optimizer: optimizer to step
        lr_scheduler: learning rate scheduler
        device_id (int): GPU device ID for this rank
        wandb: wandb object for logging
    """
    # calculate the number of steps in an epoch
    num_batches_per_epoch = len(dataset.dataloader)
    total_training_steps = num_batches_per_epoch * args.num_epochs

    # set up model, autocast, and dtypes
    model.train()
    autocast = get_autocast(args.precision)

    # set up logging
    step_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    # loop through the batches in this epoch
    for step_num, samples in tqdm(enumerate(dataset.dataloader),
        disable=args.rank != 0,
        total=total_training_steps,
        initial=epoch * num_batches_per_epoch,
    ):
    # for step_num, samples in enumerate(dataset.dataloader):
        if step_num < resume_from_step:
            # Jump to the resume step.
            continue
        data_time_m.update(time.time() - end)
        global_step = step_num + epoch * num_batches_per_epoch

        # call compute_loss_fn on each dataset; call backward before continuing
        losses_to_log = {}
        batch_metadata_to_log = {}
        # images, (input_ids, attention_mask) = samples
        # unpack the batch and move to device
        images = samples["images"]
        if not isinstance(images, list):
            images = images.to(device_id, non_blocking=True)
        input_ids = samples["input_ids"].to(device_id, non_blocking=True)
        attention_mask = samples["attention_mask"].to(device_id, non_blocking=True)
        labels = samples["labels"].to(device_id, non_blocking=True)

        # save some metadata for logging
        batch_metadata_to_log[
            f"{dataset.name}_num_tokens"
        ] = attention_mask.sum().item()
        batch_metadata_to_log[f"{dataset.name}_num_images"] = (
            (input_ids == unwrap_model(model).media_token_id).sum().item()
        )

        # forward pass
        loss = compute_loss_fn(
            model=model,
            tokenizer=tokenizer,
            images=images,
            image_size=samples['image_size'],
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            autocast=autocast,
        )
        losses_to_log["train_loss"] = loss.item()
        divided_loss = loss / args.gradient_accumulation_steps
        divided_loss.backward()

        if args.dryrun:
            del loss
            del divided_loss
            optimizer.zero_grad(set_to_none=True)
            continue


        # FIXME: Where are the special tokens added/defined?
        # if (not args.freeze_lm_embeddings) and (
        #     not args.fsdp or args.fsdp_use_orig_params
        # ):
        #     # Mask gradients for input embeddings s.t. we only update the added tokens <image> and <|endofchunk|>
        #     if args.fsdp:
        #         embed_grad = model.lang_encoder.get_input_embeddings().weight.grad
        #     else:
        #         embed_grad = (
        #             model.module.lang_encoder.get_input_embeddings().weight.grad
        #         )
        #     zero_mask = torch.zeros_like(embed_grad)
        #     zero_mask[media_token_id] = torch.ones_like(zero_mask[media_token_id])
        #     zero_mask[endofchunk_token_id] = torch.ones_like(
        #         zero_mask[endofchunk_token_id]
        #     )
        #     if args.fsdp:
        #         model.lang_encoder.get_input_embeddings().weight.grad = (
        #             embed_grad * zero_mask
        #         )
        #     else:
        #         model.module.lang_encoder.get_input_embeddings().weight.grad = (
        #             embed_grad * zero_mask
        #         )

        # clip gradient norm
        if args.fsdp:
            model.clip_grad_norm_(1.0, norm_type=2.0)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # step optimizer and log
        if (((step_num + 1) % args.gradient_accumulation_steps) == 0) or (
            step_num == num_batches_per_epoch - 1
        ):
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            # step time and reset end outside of rank 0
            step_time_m.update(time.time() - end)
            end = time.time()

            # rank 0 logging
            if args.rank == 0 and args.report_to_wandb:
                # calculate samples per second
                throughput_metrics = compute_throughput(
                    args,
                    [dataset],
                    batch_metadata_to_log,
                    step_time_m,
                )
                wandb.log(
                    {
                        "global_step": global_step,
                        "lr": optimizer.param_groups[0]["lr"],
                        **losses_to_log,
                        "data_time": data_time_m.avg,
                        "step_time": step_time_m.avg,
                        **throughput_metrics,
                    },
                    commit=True,
                )
                step_time_m.reset()
                data_time_m.reset()
        
        # dist.barrier()

        # Log loss to console
        if ((step_num + 1) % args.logging_steps == 0) and args.rank == 0:
            print(
                f"Step {step_num+1}/{num_batches_per_epoch} of epoch {epoch+1}/{args.num_epochs} complete. Losses: "
                + "// ".join([f"{k}: {v:.3f}" for k, v in losses_to_log.items()])
            )
        if ((step_num + 1) % args.checkpoint_steps == 0):
            save_checkpoint(model, optimizer, lr_scheduler, epoch, args, step=step_num)


def get_autocast(precision, cache_enabled=True):
    """
    Parses the precision argument and returns an autocast context manager.
    """
    if precision == "amp":
        return torch.cuda.amp.autocast(cache_enabled=cache_enabled)
    elif precision == "amp_bfloat16" or precision == "amp_bf16":
        return lambda: torch.cuda.amp.autocast(
            dtype=torch.bfloat16, cache_enabled=cache_enabled
        )
    else:
        return suppress


def random_seed(seed=42, rank=0):
    """Seed everything"""
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def unwrap_model(model):
    """
    Unwrap a model from a DataParallel or DistributedDataParallel wrapper.
    """
    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        return model.module
    else:
        return model


################################
# Helper functions for logging #
################################


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_throughput(
    args,
    datasets,
    batch_metadata,
    step_time_m,
):
    """
    Computes throughput metrics for logging, including samples per second and tokens per second.
    """
    log = {}
    for dataset in datasets:
        log[f"{dataset.name}_samples_per_second_per_gpu"] = (
            args.gradient_accumulation_steps * dataset.batch_size / step_time_m.val
        )
        log[f"{dataset.name}_samples_per_second"] = (
            log[f"{dataset.name}_samples_per_second_per_gpu"] * args.world_size
        )
        log[f"{dataset.name}_tokens_per_second_per_gpu"] = (
            args.gradient_accumulation_steps
            * batch_metadata[f"{dataset.name}_num_tokens"]
            / step_time_m.val
        )
        log[f"{dataset.name}_tokens_per_second"] = (
            log[f"{dataset.name}_tokens_per_second_per_gpu"] * args.world_size
        )  # this is an estimate based on rank 0
        log[f"{dataset.name}_images_per_second_per_gpu"] = (
            args.gradient_accumulation_steps
            * batch_metadata[f"{dataset.name}_num_images"]
            / step_time_m.val
        )
        log[f"{dataset.name}_images_per_second"] = (
            log[f"{dataset.name}_images_per_second_per_gpu"] * args.world_size
        )  # this is an estimate based on rank 0

    return log


####################################################
# Helper functions for checkpoint loading / saving #
####################################################


def find_most_recent_checkpoint(args):
    """
    Returns the path of the most recent checkpoint for a given run name.
    """
    checkpoint_list = glob.glob(f"{args.run_name}/checkpoint_*.pt")
    if len(checkpoint_list) == 0:
        print(f"Found no checkpoints for run {args.run_name}.")
        resume_from_checkpoint = None
    else:
        resume_from_checkpoint = sorted(
            checkpoint_list, key=lambda x: int(x.split("_")[-1].split(".")[0])
        )[-1]
        print(f"Found checkpoint {resume_from_checkpoint} for run {args.run_name}.")
    return resume_from_checkpoint


def load_checkpoint(args, model, pretrained=False):
    """
    Loads a checkpoint into the model and returns the checkpoint + epoch to resume from.
    Does not load the optimizer or learning rate checkpoints, but these are included in the returned checkpoint dict.
    """
    if pretrained:
        ckpt_path = args.pretrained
    else:
        ckpt_path = args.resume_from_checkpoint

    if args.rank == 0:
        print(f"Loading checkpoint from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    msd = checkpoint.pop("model_state_dict")
    msd = {k.replace("module.", ""): v for k, v in msd.items()}

    if 'vision_tokenizer.latents' in msd.keys():
        msd_current = model.state_dict()
        if msd_current['vision_tokenizer.latents'].shape != msd['vision_tokenizer.latents'].shape:
            msd["vision_tokenizer.latents"] = msd_current['vision_tokenizer.latents'] # Random re-init.

    # remove any module with vision_encoder in the name
    # msd = {k: v for k, v in msd.items() if "vision_encoder" not in k}

    if not pretrained:
        resume_from_epoch = checkpoint["epoch"] + 1
    else:
        resume_from_epoch = None
    
    if 'step' in checkpoint and checkpoint["step"] is not None:
        resume_from_step = checkpoint["step"] + 1
        resume_from_epoch = checkpoint["epoch"] # Resume from prev epoch at the given step.
    else:
        resume_from_step = 0

    if args.fsdp:
        FSDP.set_state_dict_type(
            model,
            **args.fsdp_checkpoint_config,
        )
    result = model.load_state_dict(msd, strict=False)
    # Print missing and unexpected keys
    print("Missing keys:", result.missing_keys)
    print("Unexpected keys:", result.unexpected_keys)
    
    return resume_from_epoch, resume_from_step, checkpoint

def filter_state_dict_to_trainable(model, state_dict):
    """
    Remove non-trainable parameters from model state dict.
    Exception: Embeddings will not be removed, even if frozen.
    This is because we need the new <image> <|endofchunk|> tokens to
    be consistent across initializations.
    """
    # first, remove frozen params
    for name, p in model.named_parameters():
        if "fsdp" in name:
            continue
        if not p.requires_grad:
            name = name.replace("._checkpoint_wrapped_module", "")
            if name in state_dict:
                del state_dict[name]
            else:
                print(f"WARNING: filtering but {name} not in state_dict")
    # second, remove additional duplicate params
    duplicate = lambda k: (
        "lang_model.old_decoder_blocks" in k
        or "lang_model.gated_cross_attn_layers" in k
    )
    filtered_dict = {
        key: value for key, value in state_dict.items() if not duplicate(key)
    }
    return filtered_dict


def save_checkpoint(model, optimizer, lr_scheduler, epoch, args, step=None):
    """
    Save training checkpoint with model, optimizer, and lr_scheduler state.
    """
    torch.cuda.empty_cache() # (Sometimes this is necessary to avoid OOM errors when saving checkpoints)

    if args.fsdp:
        FSDP.set_state_dict_type(
            model,
            **args.fsdp_checkpoint_config,
        )
        model_state = model.state_dict()
        optim_state = FSDP.optim_state_dict(model, optimizer)
    else:
        model_state = model.state_dict()
        optim_state = optimizer.state_dict()

    if args.rank == 0:
        model_state = filter_state_dict_to_trainable(model, model_state)

        if not os.path.exists(args.run_name):
            os.makedirs(args.run_name)

        checkpoint_dict = {
            "epoch": epoch,
            "step": step,
            "model_state_dict": model_state,
            "optimizer_state_dict": optim_state,
            "lr_scheduler_state_dict": lr_scheduler.state_dict(),
        }
        if args.no_save_optim_state and step is None:
            del checkpoint_dict['optimizer_state_dict']
            del checkpoint_dict['lr_scheduler_state_dict']

        if step is not None:
            save_name = f"{args.run_name}/checkpoint_{step}.pt"
        else:
            save_name = f"{args.run_name}/checkpoint_{epoch}.pt"
        print(f"Saving checkpoint to {save_name}")
        torch.save(checkpoint_dict, save_name)
        if args.report_to_wandb and args.save_checkpoints_to_wandb:
            wandb.save(f"{save_name}")

        if args.delete_previous_checkpoint:
            if epoch > 0:
                os.remove(f"{args.run_name}/checkpoint_{epoch-1}.pt")
            else:
                checkpoint_list = glob.glob(f"{args.run_name}/checkpoint_*.pt")
                if len(checkpoint_list) > 1:
                    last_checkpoint = sorted(
                        checkpoint_list, key=lambda x: int(x.split("_")[-1].split(".")[0])
                    )[0]
                    os.remove(f"{last_checkpoint}")
