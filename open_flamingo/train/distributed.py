"""
Util functions for distributed training and FSDP.
"""

import os
import torch

##################################
# SLURM setup; Credit: open_clip #
##################################

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


def is_global_master(args):
    return args.rank == 0


def is_local_master(args):
    return args.local_rank == 0


def is_master(args, local=False):
    return is_local_master(args) if local else is_global_master(args)


def is_using_horovod():
    # NOTE w/ horovod run, OMPI vars should be set, but w/ SLURM PMI vars will be set
    # Differentiating between horovod and DDP use via SLURM may not be possible, so horovod arg still required...
    ompi_vars = ["OMPI_COMM_WORLD_RANK", "OMPI_COMM_WORLD_SIZE"]
    pmi_vars = ["PMI_RANK", "PMI_SIZE"]
    if all([var in os.environ for var in ompi_vars]) or all(
        [var in os.environ for var in pmi_vars]
    ):
        return True
    else:
        return False


def is_using_distributed():
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"]) > 1
    if "SLURM_NTASKS" in os.environ:
        return int(os.environ["SLURM_NTASKS"]) > 1
    return False


def world_info_from_env():
    local_rank = 0
    for v in (
        "LOCAL_RANK",
        "MPI_LOCALRANKID",
        "SLURM_LOCALID",
        "OMPI_COMM_WORLD_LOCAL_RANK",
    ):
        if v in os.environ:
            local_rank = int(os.environ[v])
            break
    global_rank = 0
    for v in ("RANK", "PMI_RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK"):
        if v in os.environ:
            global_rank = int(os.environ[v])
            break
    world_size = 1
    for v in ("WORLD_SIZE", "PMI_SIZE", "SLURM_NTASKS", "OMPI_COMM_WORLD_SIZE"):
        if v in os.environ:
            world_size = int(os.environ[v])
            break

    return local_rank, global_rank, world_size


def init_distributed_device(args):
    # Distributed training = training on more than one GPU.
    # Works in both single and multi-node scenarios.
    args.distributed = False
    args.world_size = 1
    args.rank = 0  # global rank
    args.local_rank = 0
    if args.horovod:
        assert hvd is not None, "Horovod is not installed"
        hvd.init()
        args.local_rank = int(hvd.local_rank())
        args.rank = hvd.rank()
        args.world_size = hvd.size()
        args.distributed = True
        os.environ["LOCAL_RANK"] = str(args.local_rank)
        os.environ["RANK"] = str(args.rank)
        os.environ["WORLD_SIZE"] = str(args.world_size)
    elif is_using_distributed():
        if "SLURM_PROCID" in os.environ:
            # DDP via SLURM
            args.local_rank, args.rank, args.world_size = world_info_from_env()
            # SLURM var -> torch.distributed vars in case needed
            os.environ["LOCAL_RANK"] = str(args.local_rank)
            os.environ["RANK"] = str(args.rank)
            os.environ["WORLD_SIZE"] = str(args.world_size)
            torch.distributed.init_process_group(
                backend=args.dist_backend,
                init_method=args.dist_url,
                world_size=args.world_size,
                rank=args.rank,
            )
        else:
            # DDP via torchrun, torch.distributed.launch
            args.local_rank, _, _ = world_info_from_env()
            torch.distributed.init_process_group(
                backend=args.dist_backend, init_method=args.dist_url
            )
            args.world_size = torch.distributed.get_world_size()
            args.rank = torch.distributed.get_rank()
        args.distributed = True
    else:
        # needed to run on single gpu
        torch.distributed.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=1,
            rank=0,
        )

    if torch.cuda.is_available():
        if args.distributed and not args.no_set_device_rank:
            device = "cuda:%d" % args.local_rank
        else:
            device = "cuda:0"
        torch.cuda.set_device(device)
    else:
        device = "cpu"
    args.device = device
    device = torch.device(device)
    return device


#####################################
# FSDP util functions #
#####################################


def get_fsdp_mixed_precision_policy(
    precision: str,
    reduce_param_precision=False,
    reduce_communication_precision=True,
    reduce_buffer_precision=True,
):
    """
    Returns the FSDP mixed precision policy for a given precision.
    """
    if "bfloat16" in precision or "bf16" in precision:
        cast_dtype = torch.bfloat16
    elif precision == "fp16":
        cast_dtype = torch.float16
    else:
        cast_dtype = torch.float32

    if cast_dtype == torch.float32:
        return None

    from torch.distributed.fsdp import MixedPrecision

    return MixedPrecision(
        param_dtype=cast_dtype if reduce_param_precision else torch.float32,
        reduce_dtype=cast_dtype if reduce_communication_precision else torch.float32,
        buffer_dtype=cast_dtype if reduce_buffer_precision else torch.float32,
    )


def get_fsdp_config(
    args,
    device_id,
):
    """
    Return kwargs for FSDP wrapper.
    This includes some hard-coded settings.
    """
    # init MixedPrecision
    mp_policy = get_fsdp_mixed_precision_policy(
        args.precision,
        reduce_param_precision=False,
        reduce_communication_precision=True,
        reduce_buffer_precision=True,
    )

    # init FSDP
    from torch.distributed.fsdp import (
        ShardingStrategy,
        BackwardPrefetch,
        CPUOffload
    )

    if args.fsdp_sharding_strategy == "full":
        sharding_strategy = ShardingStrategy.FULL_SHARD
    elif args.fsdp_sharding_strategy == "hybrid":
        sharding_strategy = ShardingStrategy.HYBRID_SHARD
    elif args.fsdp_sharding_strategy == "shard_grad_op":
        sharding_strategy = ShardingStrategy.SHARD_GRAD_OP
    elif args.fsdp_sharding_strategy == "hybrid_shard_grad_op":
        sharding_strategy = ShardingStrategy._HYBRID_SHARD_ZERO2
    elif args.fsdp_sharding_strategy == "no_shard":
        sharding_strategy = ShardingStrategy.NO_SHARD
    else:
        raise ValueError(
            f"Invalid sharding strategy: {args.fsdp_sharding_strategy}. Supported: full, hybrid, shard_grad_op, hybrid_shard_grad_op, no_shard"
        )
    if args.cpu_offload_gradients:
        cpu_offload = CPUOffload(offload_params=True)
    else:
        cpu_offload = None

    return dict(
        cpu_offload=cpu_offload,
        device_id=device_id,
        sync_module_states=True,  # broadcast loaded ckpt from rank 0 -> all ranks
        sharding_strategy=sharding_strategy,
        use_orig_params=True,
        mixed_precision=mp_policy,
        forward_prefetch=True,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        limit_all_gathers=True,
    )


def get_fsdp_checkpoint_config(args):
    """
    Return kwargs for FSDP checkpointing.
    """
    from torch.distributed.fsdp import (
        FullStateDictConfig,
        StateDictType,
    )
    from torch.distributed.fsdp.api import FullOptimStateDictConfig

    # to avoid GPU OOM when loading/saving ckpts, load/save on CPU
    # this is slow
    return dict(
        state_dict_type=StateDictType.FULL_STATE_DICT,
        state_dict_config=FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
        optim_state_dict_config=FullOptimStateDictConfig(
            rank0_only=True, offload_to_cpu=True
        ),
    )
