import argparse

from config import OliveConfig


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
    args = parse_args()

    configs = OliveConfig(args)

    pass

if __name__ == '__main__':
    main()