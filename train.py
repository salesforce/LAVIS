import argparse
import os

from omegaconf import OmegaConf
from runners.runner_base import Runner

import tasks

from common.registry import registry
from utils.config import Config
from utils.logger import setup_logger

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
    # model = registry.get_model_class("blip_enc_dec").build_model()

    root_dir = os.getcwd()
    default_cfg = OmegaConf.load(os.path.join(root_dir, "configs/default.yaml"))

    registry.register_path("library_root", root_dir)
    registry.register_path("cache_root", default_cfg.env.cache_root)

    setup_logger()

    cfg = Config(parse_args())
    if utils.is_main_process():
        cfg.pretty_print()

    utils.init_distributed_mode(cfg.run_cfg)

    task = tasks.setup_task(cfg)

    datasets = task.build_datasets(cfg)

    model = task.build_model(cfg)

    runner = Runner(cfg=cfg.run_cfg, task=task, model=model, datasets=datasets)
    runner.train()


if __name__ == "__main__":
    main()
