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
    parser = argparse.ArgumentParser(description='Training')

    # [TODO] Might not be necessary to have three seperate files.
    # Consider relocating to one file.
    parser.add_argument('--cfg-run', required=True, help='runner-specific config file path.')
    parser.add_argument('--cfg-data', required=True, help='dataset-specific config file path.')
    parser.add_argument('--cfg-model', required=True, help='model-specific config file path.')
    
    parser.add_argument('--options',
        nargs='+',
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file (deprecate), '
        'change to --cfg-options instead.')

    # TODO to relocate to run configuration
    # parser.add_argument('--config', default='./temp/caption_coco.yaml')
    # parser.add_argument('--output_dir', default='output/Caption_coco')        
    # parser.add_argument('--evaluate', action='store_true')    
    # parser.add_argument('--device', default='cuda')
    # parser.add_argument('--seed', default=42, type=int)
    # parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    # parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    # parser.add_argument('--distributed', default=True, type=bool)

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


# def train(model, data_loader, optimizer, epoch, device):
#     # train
#     model.train()  
    
#     metric_logger = utils.MetricLogger(delimiter="  ")
#     metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
#     metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
#     header = 'Train Caption Epoch: [{}]'.format(epoch)
#     print_freq = 50

#     for i, (image, caption, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
#         image = image.to(device)       
        
#         loss = model(image, caption)      
        
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()    
        
#         metric_logger.update(loss=loss.item())
#         metric_logger.update(lr=optimizer.param_groups[0]["lr"])

#     # gather the stats from all processes
#     metric_logger.synchronize_between_processes()
#     print("Averaged stats:", metric_logger.global_avg())     
#     return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}  

# import torch
# @torch.no_grad()
# def evaluate(model, data_loader, device, config):
#     # evaluate
#     model.eval() 
    
#     metric_logger = utils.MetricLogger(delimiter="  ")
#     header = 'Caption generation:'
#     print_freq = 10

#     result = []
#     for image, image_id in metric_logger.log_every(data_loader, print_freq, header): 
        
#         image = image.to(device)       
        
#         samples = {"vis_data": image, "id": image_id}

#         captions = model.generate(samples, use_nucleus_sampling=False, num_beams=config['num_beams'], max_length=config['max_length'], 
#                                   min_length=config['min_length'])
#         # captions = model.generate(image, sample=False, num_beams=config['num_beams'], max_length=config['max_length'], 
#         #                           min_length=config['min_length'])
        
#         for caption, img_id in zip(captions, image_id):
#             result.append({"image_id": img_id.item(), "caption": caption})
  
#     return result


def main():

    # model = registry.get_model_class("blip_enc_dec").build_model()
    # import pdb; pdb.set_trace()

    setup_path()
    setup_logger()

    args = parse_args()

    cfg = Config(args)
    # cfg.pretty_print()

    task = tasks.setup_task(cfg)

    datasets = task.build_datasets(cfg)
    # FIXME to support dataloaders for multiple datasets
    datasets = datasets[list(datasets.keys())[0]]

    model = task.build_model(cfg)

    utils.init_distributed_mode(cfg.get_runner_config())

    runner = Runner(cfg=cfg.get_runner_config(), task=task, model=model, datasets=datasets)
    runner.train_loop()

    # ======================================
    # code below has to be reorganized into a trainer.
    # [TODO] to remove
    # import torch
    # import random, time, json, datetime
    # import numpy as np
    # import torch.backends.cudnn as cudnn
    # import torch.distributed as dist

    # args = parser.parse_args()

    # utils.init_distributed_mode(args)

    # device = torch.device(args.device)

    # fix the seed for reproducibility
    # seed = args.seed + utils.get_rank()
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    # random.seed(seed)
    # cudnn.benchmark = True

    # import yaml
    # from pathlib import Path
    # args.result_dir = os.path.join(args.output_dir, 'result')
    # Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    # Path(args.result_dir).mkdir(parents=True, exist_ok=True)

    # config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    #### Dataset ####
    # print("Creating captioning dataset")
    # datasets = datasets['coco_caption']
    # train_dataset, val_dataset, test_dataset = datasets['train'], datasets['val'], datasets['test']
        # create_dataset('caption_coco', config)

    # if args.distributed:
    #     num_tasks = utils.get_world_size()
    #     global_rank = utils.get_rank()
    #     samplers = utils.create_sampler([train_dataset, val_dataset, test_dataset], [True, False, False], num_tasks,
    #                               global_rank)
    # else:
    #     samplers = [None, None, None]

    # train_loader, val_loader, test_loader = utils.create_loader([train_dataset, val_dataset, test_dataset], samplers,
    #                                                       batch_size=[config['batch_size']] * 3,
    #                                                       num_workers=[4, 4, 4],
    #                                                       is_trains=[True, False, False],
    #                                                       collate_fns=[None, None, None])

    #### Model ####
    # print("Creating model")
    # model = task.build_model(cfg)
    # model = blip_decoder(pretrained=config['pretrained'], image_size=config['image_size'], vit=config['vit'],
    #                      vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'],
    #                      prompt=config['prompt'])


    # model_without_ddp = model
    # if args.distributed:
    #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    #     model_without_ddp = model.module

    # optimizer = torch.optim.AdamW(params=model.parameters(), lr=float(config['init_lr']),
    #                               weight_decay=float(config['weight_decay']))

    # best = 0
    # best_epoch = 0

    # # print("Start training")
    # start_time = time.time()
    # for epoch in range(0, config['max_epoch']):
    #     if not args.evaluate:
    #         if args.distributed:
    #             train_loader.sampler.set_epoch(epoch)

    #         utils.cosine_lr_schedule(optimizer, epoch, int(config['max_epoch']), float(config['init_lr']), float(config['min_lr']))

    #         train_stats = train(model, train_loader, optimizer, epoch, device)

    #     val_result = evaluate(model_without_ddp, val_loader, device, config)
    #     val_result_file = utils.save_result(val_result, args.result_dir, 'val_epoch%d' % epoch,
    #                                   remove_duplicate='image_id')

    #     test_result = evaluate(model_without_ddp, test_loader, device, config)
    #     test_result_file = utils.save_result(test_result, args.result_dir, 'test_epoch%d' % epoch,
    #                                    remove_duplicate='image_id')

    #     if utils.is_main_process():
    #         coco_val = utils.coco_caption_eval(config['coco_gt_root'], val_result_file, 'val')
    #         coco_test = utils.coco_caption_eval(config['coco_gt_root'], test_result_file, 'test')

    #         if args.evaluate:
    #             log_stats = {**{f'val_{k}': v for k, v in coco_val.eval.items()},
    #                          **{f'test_{k}': v for k, v in coco_test.eval.items()},
    #                          }
    #             with open(os.path.join(args.output_dir, "evaluate.txt"), "a") as f:
    #                 f.write(json.dumps(log_stats) + "\n")
    #         else:
    #             save_obj = {
    #                 'model': model_without_ddp.state_dict(),
    #                 'optimizer': optimizer.state_dict(),
    #                 'config': config,
    #                 'epoch': epoch,
    #             }

    #             if coco_val.eval['CIDEr'] + coco_val.eval['Bleu_4'] > best:
    #                 best = coco_val.eval['CIDEr'] + coco_val.eval['Bleu_4']
    #                 best_epoch = epoch
    #                 torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth'))

    #             log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
    #                          **{f'val_{k}': v for k, v in coco_val.eval.items()},
    #                          **{f'test_{k}': v for k, v in coco_test.eval.items()},
    #                          'epoch': epoch,
    #                          'best_epoch': best_epoch,
    #                          }
    #             with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
    #                 f.write(json.dumps(log_stats) + "\n")

    #     if args.evaluate:
    #         break
    #     dist.barrier()

    # total_time = time.time() - start_time
    # total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    # print('Training time {}'.format(total_time_str))




    # 3. task.build_model(cfg.model)
    # 4. criterion = task.build_criterion(cfg.run.criterion)
    # 5. optimizer = build_optimizer(cfg.run.optimizer)
    # 6. runner = Runner(cfg, task, model, criterion, optimizer)

if __name__ == '__main__':
    main()