import os

import torch
import random, time, json, datetime
import numpy as np
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import utils.blip_utils as utils

import yaml
from pathlib import Path
from common.registry import registry


class Runner():
    def __init__(self, cfg, task, model, datasets):
        self.config = cfg

        self.task = task
        self.model = model

        # TODO handle cases where some splits are missing.
        self.datasets = datasets
        self.train_dataset = datasets.get('train', None)
        self.val_dataset = datasets.get('val', None)
        self.test_dataset = datasets.get('test', None)

        self.setup_seeds()
        self.setup_device()
        self.setup_output_dir()

        self.setup_model()

        self.setup_optimizer()

        self.setup_dataloaders()


    def setup_seeds(self):
        seed = self.config.seed + utils.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        cudnn.benchmark = True

    def setup_device(self):
        self.device = torch.device(self.config.device)

    def setup_model(self):
        self.model = self.model.to(self.device)

        # setup_dist_model()
        self.model_without_ddp = self.model
        if self.config.distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.config.gpu])
            self.model_without_ddp = self.model.module

    def setup_output_dir(self):
        lib_root = Path(registry.get_path("library_root"))

        output_dir = lib_root / 'output'
        result_dir = output_dir / 'result'

        output_dir.mkdir(parents=True, exist_ok=True)
        result_dir.mkdir(parents=True, exist_ok=True)

        registry.register_path("result_dir", str(result_dir))
        registry.register_path("output_dir", str(output_dir))

        self.result_dir = result_dir
        self.output_dir = output_dir

    def setup_optimizer(self):
        # TODO make optimizer class and configurations
        self.optimizer = torch.optim.AdamW(
            params=self.model.parameters(), 
            lr=float(self.config.init_lr),
            weight_decay=float(self.config.weight_decay)
        )

    @property
    def cuda_enabled(self):
        return self.device.type == "cuda"

    def setup_dataloaders(self):
        # TODO this method has to be rewritten
        # TODO make number of workers configurable
        if self.config.distributed:
            num_tasks = utils.get_world_size()
            global_rank = utils.get_rank()
            samplers = utils.create_sampler(
                [self.train_dataset, self.val_dataset, self.test_dataset], 
                [True, False, False], 
                num_tasks, 
                global_rank
            )
        else:
            samplers = [None, None, None]

        # TODO handle cases where some splits are missing.
        train_loader, val_loader, test_loader = utils.create_loader(
            [self.train_dataset, self.val_dataset, self.test_dataset], 
            samplers,
            batch_size=[self.config.batch_size] * 3,
            num_workers=[4, 4, 4],
            is_trains=[True, False, False],
            collate_fns=[None, None, None]
        )

        # TODO handle cases where some splits are missing.
        self.dataloaders = {
            "train": train_loader,
            "val": val_loader,
            "test": test_loader
        } 

    def train_loop(self):
        best = 0
        best_epoch = 0

        # print("Start training")
        start_time = time.time()
        for epoch in range(0, self.config.max_epoch):
            # if not self.args.evaluate:
            #     if self.args.distributed:
            #         self.train_loader.sampler.set_epoch(epoch)

            #     utils.cosine_lr_schedule(self.optimizer, epoch, int(self.config['max_epoch']), float(self.config['init_lr']), float(self.config['min_lr']))

            #     train_stats = self.train(epoch)

            # for split_name in val_split_names:
            val_result = self.validation(split_name='val')
            self.task.on_finish_validation(val_result=val_result, split_name='val', epoch=epoch)

            # val_result_file = utils.save_result(val_result, self.result_dir, 'val_epoch%d' % epoch,
            #                             remove_duplicate='image_id')
            # val_result_file = utils.save_result(val_result, self.result_dir, 'val_epoch%d' % epoch, remove_duplicate='id')

            test_result = self.validation(split_name='test')
            self.task.on_finish_validation(val_result=test_result, split_name='test', epoch=epoch)
            # test_result_file = utils.save_result(test_result, self.result_dir, 'test_epoch%d' % epoch,
            #                             remove_duplicate='image_id')
            # test_result_file = utils.save_result(test_result, self.result_dir, 'test_epoch%d' % epoch, remove_duplicate='id')


            # if utils.is_main_process():
            #     coco_val = utils.coco_caption_eval(self.config['coco_gt_root'], val_result_file, 'val')
            #     coco_test = utils.coco_caption_eval(self.config['coco_gt_root'], test_result_file, 'test')

            #     if self.args.evaluate:
            #         log_stats = {**{f'val_{k}': v for k, v in coco_val.eval.items()},
            #                     **{f'test_{k}': v for k, v in coco_test.eval.items()},
            #                     }
            #         with open(os.path.join(self.args.output_dir, "evaluate.txt"), "a") as f:
            #             f.write(json.dumps(log_stats) + "\n")
            #     else:
            #         save_obj = {
            #             'model': self.model_without_ddp.state_dict(),
            #             'optimizer': self.optimizer.state_dict(),
            #             'config': self.config,
            #             'epoch': epoch,
            #         }

            #         if coco_val.eval['CIDEr'] + coco_val.eval['Bleu_4'] > best:
            #             best = coco_val.eval['CIDEr'] + coco_val.eval['Bleu_4']
            #             best_epoch = epoch
            #             torch.save(save_obj, os.path.join(self.output_dir, 'checkpoint_best.pth'))

            #         log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
            #                     **{f'val_{k}': v for k, v in coco_val.eval.items()},
            #                     **{f'test_{k}': v for k, v in coco_test.eval.items()},
            #                     'epoch': epoch,
            #                     'best_epoch': best_epoch,
            #                     }
            #         with open(os.path.join(self.output_dir, "log.txt"), "a") as f:
            #             f.write(json.dumps(log_stats) + "\n")

            if self.config.evaluate:
                break
            dist.barrier()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))


    # def train(self, epoch):
    #     # train
    #     self.model.train()  
        
    #     metric_logger = utils.MetricLogger(delimiter="  ")
    #     metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    #     metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    #     header = 'Train Caption Epoch: [{}]'.format(epoch)
    #     print_freq = 50

    #     for i, (image, caption, _) in enumerate(metric_logger.log_every(self.train_loader, print_freq, header)):
    #         image = image.to(self.device)
            
    #         loss = self.model(image, caption)      
            
    #         self.optimizer.zero_grad()
    #         loss.backward()
    #         self.optimizer.step()    
            
    #         metric_logger.update(loss=loss.item())
    #         metric_logger.update(lr=self.optimizer.param_groups[0]["lr"])

    #     # gather the stats from all processes
    #     metric_logger.synchronize_between_processes()
    #     print("Averaged stats:", metric_logger.global_avg())     
    #     return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}  

    @torch.no_grad()
    def validation(self, split_name):
        # TODO In validation, you need to compute loss as well as metrics
        model = self.model_without_ddp
        model.eval()

        data_loader = self.dataloaders.get(split_name, None)

        assert data_loader, "data_loader for split {} is None.".format(split_name)
        
        # TODO doesn't look like a good place to define logger
        # Possibly called multiple times on different splits.
        metric_logger = utils.MetricLogger(delimiter="  ")
        header = 'Caption generation:'
        # TODO make it configurable
        print_freq = 10

        results = []

        for samples in metric_logger.log_every(data_loader, print_freq, header): 
            samples = self._prepare_sample(samples)
            
            eval_output = self.task.valid_step(model=model, samples=samples)
            results.extend(eval_output)
    
        return results

    def _prepare_sample(self, samples):
        if self.cuda_enabled:
            samples = utils.move_to_cuda(samples)
        
        # TODO fp16 support

        return samples