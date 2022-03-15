import argparse
import os

from omegaconf import OmegaConf

import tasks

from utils.config import Config
from utils.logger import setup_logger
from datasets.builders import *
from common.registry import registry
from tasks import *


def parse_args():
    parser = argparse.ArgumentParser(description='Training')

    parser.add_argument(
        '--cfg-run', required=True, help='runner-specific config file path.')
    parser.add_argument(
        '--cfg-data', required=True, help='dataset-specific config file path.')
    parser.add_argument(
        '--cfg-model', required=True, help='model-specific config file path.')
    
    parser.add_argument(
        '--options',
        nargs='+',
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file (deprecate), '
        'change to --cfg-options instead.')

    # parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    # if 'LOCAL_RANK' not in os.environ:
    #     os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def setup_path():
    root_dir = os.getcwd()
    default_cfg = OmegaConf.load(os.path.join(root_dir, 'configs/default.yaml'))

    registry.register_path('library_root', root_dir)
    registry.register_path('cache_root', default_cfg.env.cache_root)


def main():
    setup_path()
    setup_logger()

    args = parse_args()

    cfg = Config(args)
    cfg.pretty_print()

    task = tasks.setup_task(cfg)
    # [TODO]
    datasets = task.build_datasets(cfg) 

    # 3. task.build_model(cfg.model)
    # 4. criterion = task.build_criterion(cfg.run.criterion)
    # 5. optimizer = build_optimizer(cfg.run.optimizer)
    # 6. runner = Runner(cfg, task, model, criterion, optimizer)

if __name__ == '__main__':
    main()