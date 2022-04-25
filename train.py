import argparse
import os

from omegaconf import OmegaConf
from runners.runner_base import Runner

import tasks

from common.registry import registry
from utils.config import Config

# imports modules for registration
from datasets.builders import *
from tasks import *
from processors import *
from models import *

import utils.blip_utils as utils


def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )

    args = parser.parse_args()
    # if 'LOCAL_RANK' not in os.environ:
    #     os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    # allow auto-dl completes on main process without timeout when using NCCL backend.
    os.environ['NCCL_BLOCKING_WAIT'] = "1"

    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
    job_id = utils.now()

    root_dir = os.getcwd()
    default_cfg = OmegaConf.load(os.path.join(root_dir, "configs/default.yaml"))

    registry.register_path("library_root", root_dir)
    registry.register_path("cache_root", default_cfg.env.cache_root)

    cfg = Config(parse_args())

    utils.init_distributed_mode(cfg.run_cfg)
    # set after init_distributed_mode() to only log on master.
    utils.setup_logger()

    cfg.pretty_print()

    task = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)
    model = task.build_model(cfg)

    runner = Runner(
        cfg=cfg,
        job_id=job_id,
        task=task,
        model=model,
        datasets=datasets
    )
    runner.train()


if __name__ == "__main__":
    main()
