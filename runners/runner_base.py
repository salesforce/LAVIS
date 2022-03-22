import os

import torch
import random, time, json, datetime
import numpy as np
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import utils.blip_utils as utils

import yaml
from pathlib import Path


class Runner():
    def __init__(self, args, task, model, datasets):
        self.args = args

        self.task = task
        self.model = model

        self.device = torch.device('cuda')
        self.model = model.to(self.device)

        self.train_dataset = datasets['train']
        self.val_dataset = datasets['val']
        self.test_dataset = datasets['test']


        args.result_dir = os.path.join(args.output_dir, 'result')
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        Path(args.result_dir).mkdir(parents=True, exist_ok=True)

        self.config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

        self.set_seeds()
        self.setup_dist_model()
        self.setup_optimizer()
        self.setup_data_loaders()


    def setup_dist_model(self):
        self.model_without_ddp = self.model
        if self.args.distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.args.gpu])
            self.model_without_ddp = self.model.module


    def setup_optimizer(self):
        self.optimizer = torch.optim.AdamW(
            params=self.model.parameters(), 
            lr=float(self.config['init_lr']),
            weight_decay=float(self.config['weight_decay']
            )
        )


    def setup_data_loaders(self):
        if self.args.distributed:
            num_tasks = utils.get_world_size()
            global_rank = utils.get_rank()
            samplers = utils.create_sampler([self.train_dataset, self.val_dataset, self.test_dataset], 
                                            [True, False, False], num_tasks, global_rank)
        else:
            samplers = [None, None, None]

        self.train_loader, self.val_loader, self.test_loader = utils.create_loader([self.train_dataset, self.val_dataset, self.test_dataset], samplers,
                                                            batch_size=[self.config['batch_size']] * 3,
                                                            num_workers=[4, 4, 4],
                                                            is_trains=[True, False, False],
                                                            collate_fns=[None, None, None])

    def set_seeds(self):
        seed = self.args.seed + utils.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        cudnn.benchmark = True

    def train_loop(self):
        best = 0
        best_epoch = 0

        # print("Start training")
        start_time = time.time()
        for epoch in range(0, self.config['max_epoch']):
            # if not self.args.evaluate:
            #     if self.args.distributed:
            #         self.train_loader.sampler.set_epoch(epoch)

            #     utils.cosine_lr_schedule(self.optimizer, epoch, int(self.config['max_epoch']), float(self.config['init_lr']), float(self.config['min_lr']))

            #     train_stats = self.train(epoch)

            val_result = self.evaluate(self.val_loader)
            val_result_file = utils.save_result(val_result, self.args.result_dir, 'val_epoch%d' % epoch,
                                        remove_duplicate='image_id')

            test_result = self.evaluate(self.test_loader)
            test_result_file = utils.save_result(test_result, self.args.result_dir, 'test_epoch%d' % epoch,
                                        remove_duplicate='image_id')

            if utils.is_main_process():
                coco_val = utils.coco_caption_eval(self.config['coco_gt_root'], val_result_file, 'val')
                coco_test = utils.coco_caption_eval(self.config['coco_gt_root'], test_result_file, 'test')

                if self.args.evaluate:
                    log_stats = {**{f'val_{k}': v for k, v in coco_val.eval.items()},
                                **{f'test_{k}': v for k, v in coco_test.eval.items()},
                                }
                    with open(os.path.join(self.args.output_dir, "evaluate.txt"), "a") as f:
                        f.write(json.dumps(log_stats) + "\n")
                else:
                    save_obj = {
                        'model': self.model_without_ddp.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'config': self.config,
                        'epoch': epoch,
                    }

                    if coco_val.eval['CIDEr'] + coco_val.eval['Bleu_4'] > best:
                        best = coco_val.eval['CIDEr'] + coco_val.eval['Bleu_4']
                        best_epoch = epoch
                        torch.save(save_obj, os.path.join(self.args.output_dir, 'checkpoint_best.pth'))

                    log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                                **{f'val_{k}': v for k, v in coco_val.eval.items()},
                                **{f'test_{k}': v for k, v in coco_test.eval.items()},
                                'epoch': epoch,
                                'best_epoch': best_epoch,
                                }
                    with open(os.path.join(self.args.output_dir, "log.txt"), "a") as f:
                        f.write(json.dumps(log_stats) + "\n")

            if self.args.evaluate:
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
    def evaluate(self, data_loader):
        self.model_without_ddp.eval() 
        
        metric_logger = utils.MetricLogger(delimiter="  ")
        header = 'Caption generation:'
        print_freq = 10

        result = []
        for image, image_id in metric_logger.log_every(data_loader, print_freq, header): 
            
            image = image.to(self.device)       
            
            samples = {"vis_data": image, "id": image_id}

            captions = self.model_without_ddp.generate(samples, 
                                    use_nucleus_sampling=False, 
                                    num_beams=self.config['num_beams'], 
                                    max_length=self.config['max_length'], 
                                    min_length=self.config['min_length']
                                )
            # captions = model.generate(image, sample=False, num_beams=config['num_beams'], max_length=config['max_length'], 
            #                           min_length=config['min_length'])
            
            for caption, img_id in zip(captions, image_id):
                result.append({"image_id": img_id.item(), "caption": caption})
    
        return result