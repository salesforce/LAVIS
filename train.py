import argparse

from utils.config import Config
from utils.logger import setup_logger
from datasets.builders import *


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


def main():
    setup_logger()

    args = parse_args()

    cfg = Config(args)
    cfg.pretty_print()

    # [TODO]
    # 1. task = BaseTask().setup_taks(cfg.run.task)
    # 2. task.build_model(cfg.model)
    # 3. datasets = task.build_datasets(cfg.datasets) 
    # 4. criterion = task.build_criterion(cfg.run.criterion)
    # 5. optimizer = build_optimizer(cfg.run.optimizer)
    # 6. runner = Runner(cfg, task, model, criterion, optimizer)

if __name__ == '__main__':
    main()