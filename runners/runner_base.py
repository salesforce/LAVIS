import datetime
import logging
import os
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from tasks.retrieval import RetrievalTask
import utils.blip_utils as utils
from common.registry import registry
from torch.utils.data import DataLoader


class Runner:
    def __init__(self, cfg, task, model, datasets):
        self.config = cfg

        self.task = task
        self.datasets = datasets

        self._model = model

        self._wrapped_model = None
        self._device = None
        self._optimizer = None
        self._dataloaders = None

        self.setup_seeds()
        self.setup_output_dir()

    def setup_seeds(self):
        seed = self.config.seed + utils.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        cudnn.benchmark = True

    @property
    def device(self):
        if self._device is None:
            self._device = torch.device(self.config.device)

        return self._device

    @property
    def use_distributed(self):
        return self.config.distributed

    @property
    def model(self):
        if self._model.device != self.device:
            self._model = self._model.to(self.device)

            if self.use_distributed:
                if self._wrapped_model is None:
                    self._wrapped_model = torch.nn.parallel.DistributedDataParallel(
                        self._model, device_ids=[self.config.gpu]
                    )
            else:
                self._wrapped_model = self._model

        return self._wrapped_model

    @property
    def model_without_ddp(self):
        if self.use_distributed:
            return self.model.module
        else:
            return self.model

    @property
    def optimizer(self):
        # TODO make optimizer class and configurations
        if self._optimizer is None:
            self._optimizer = torch.optim.AdamW(
                params=self.model.parameters(),
                lr=float(self.config.init_lr),
                weight_decay=float(self.config.weight_decay),
            )

        return self._optimizer

    @property
    def dataloaders(self):
        if self._dataloaders is None:

            split_names = sorted(self.datasets.keys())

            datasets = [self.datasets[split] for split in split_names]
            is_train = [split in self.train_splits for split in split_names]

            if self.use_distributed:
                samplers = utils.create_sampler(
                    datasets=datasets,
                    shuffles=is_train,
                    num_tasks=utils.get_world_size(),
                    global_rank=utils.get_rank(),
                )
            else:
                samplers = [None] * len(self.datasets)

            dataloaders = utils.create_loader(
                datasets=datasets,
                samplers=samplers,
                batch_size=[
                    self.config.batch_size_train
                    if split == "train"
                    else self.config.batch_size_eval
                    for split in split_names
                ],
                num_workers=[self.config.num_workers] * len(datasets),
                is_trains=is_train,
                collate_fns=[
                    getattr(dataset, "collater", None) for dataset in datasets
                ],
            )

            self._dataloaders = {k: v for k, v in zip(split_names, dataloaders)}

        return self._dataloaders

    @property
    def cuda_enabled(self):
        return self.device.type == "cuda"

    @property
    def max_epoch(self):
        return int(self.config.max_epoch)

    @property
    def init_lr(self):
        return float(self.config.init_lr)

    @property
    def min_lr(self):
        return float(self.config.min_lr)

    @property
    def valid_splits(self):
        valid_splits = self.config.get("valid_splits", [])

        if len(valid_splits) == 0:
            logging.warning("No validation splits found.")

        return valid_splits

    @property
    def test_splits(self):
        test_splits = self.config.get("test_splits", [])

        return test_splits

    @property
    def train_splits(self):
        train_splits = self.config.get("train_splits", [])

        assert len(train_splits) > 0, "Empty train splits."
        return train_splits

    @property
    def evaluate_only(self):
        return self.config.evaluate

    @property
    def train_loader(self):
        train_loader = self.dataloaders["train"]

        assert isinstance(train_loader, DataLoader)
        return train_loader

    def setup_output_dir(self):
        lib_root = Path(registry.get_path("library_root"))

        output_dir = lib_root / self.config.output_dir / utils.now()
        result_dir = output_dir / "result"

        output_dir.mkdir(parents=True, exist_ok=True)
        result_dir.mkdir(parents=True, exist_ok=True)

        registry.register_path("result_dir", str(result_dir))
        registry.register_path("output_dir", str(output_dir))

        self.result_dir = result_dir
        self.output_dir = output_dir

    def train(self):
        start_time = time.time()
        best_agg_metric = 0
        for cur_epoch in range(0, self.max_epoch):

            if not self.evaluate_only:
                logging.info("Start training")
                if self.use_distributed:
                    self.train_loader.sampler.set_epoch(cur_epoch)

                # lr_scheduler.before_epoch()
                utils.cosine_lr_schedule(
                    optimizer=self.optimizer,
                    epoch=cur_epoch,
                    max_epoch=self.max_epoch,
                    init_lr=self.init_lr,
                    min_lr=self.min_lr,
                )

                train_stats = self.train_epoch(cur_epoch)

                if utils.is_main_process():
                    self.log_stats(split_name="train", stats=train_stats)

            if len(self.valid_splits) > 0:
                for split_name in self.valid_splits:
                    val_result = self.validate(split_name=split_name)
                    val_log = self.task.after_validation(
                        val_result=val_result,
                        split_name=split_name,
                        epoch=cur_epoch
                    )

                    if utils.is_main_process():
                        agg_metrics = val_log["agg_metrics"]
                        if agg_metrics > best_agg_metric and split_name == "val":
                            # best_epoch = cur_epoch
                            self.save_checkpoint(cur_epoch, is_best=True)
                            self.log_stats(val_log, split_name)
            else:
                if not self.evaluate_only:
                    self.save_checkpoint(cur_epoch, is_best=False)

            if self.evaluate_only:
                break
            dist.barrier()

        if len(self.test_splits) > 0:
            for split_name in self.test_splits:
                test_result = self.validate(split_name=split_name)
                test_log = self.task.after_validation(
                    val_result=test_result,
                    split_name=split_name,
                    epoch=cur_epoch,
                    result_dir=self.result_dir,
                )

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logging.info("Training time {}".format(total_time_str))

    def train_epoch(self, epoch):
        def add_vars_to_samples():
            samples["epoch"] = epoch
            samples["num_iters_per_epoch"] = len(self.train_loader)
            samples["iters"] = i

        # train
        self.model.train()

        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter(
            "lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}")
        )
        metric_logger.add_meter(
            "loss", utils.SmoothedValue(window_size=1, fmt="{value:.4f}")
        )
        header = "Train Epoch: [{}]".format(epoch)
        print_freq = 50

        for i, samples in enumerate(
            metric_logger.log_every(self.train_loader, print_freq, header)
        ):
            samples = self._prepare_sample(samples)

            add_vars_to_samples()
            loss = self.task.train_step(model=self.model, samples=samples)

            # after_train_step()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=self.optimizer.param_groups[0]["lr"])

        # after train_epoch()
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        logging.info("Averaged stats: " + str(metric_logger.global_avg()))
        return {
            k: "{:.3f}".format(meter.global_avg)
            for k, meter in metric_logger.meters.items()
        }

    @torch.no_grad()
    def validate(self, split_name):
        # TODO In validation, you need to compute loss as well as metrics
        model = self.model_without_ddp
        model.eval()

        data_loader = self.dataloaders.get(split_name, None)

        assert data_loader, "data_loader for split {} is None.".format(split_name)

        # TODO doesn't look like a good place to define logger
        # Possibly called multiple times on different splits.
        metric_logger = utils.MetricLogger(delimiter="  ")
        header = "Validation"
        # TODO make it configurable
        print_freq = 10

        results = []

        for samples in metric_logger.log_every(data_loader, print_freq, header):
            samples = self._prepare_sample(samples)

            eval_output = self.task.valid_step(model=model, samples=samples)
            results.extend(eval_output)

        dist.barrier()
        return results

    @torch.no_grad()
    def validate_retrieval(self, split_name):
        model = self.model_without_ddp
        model.eval()

        data_loader = self.dataloaders.get(split_name, None)

        assert data_loader, "data_loader for split {} is None.".format(split_name)

    def _prepare_sample(self, samples):
        if self.cuda_enabled:
            samples = utils.move_to_cuda(samples)

        # TODO fp16 support

        return samples

    def save_checkpoint(self, cur_epoch, is_best=False):
        save_obj = {
            "model": self.model_without_ddp.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            # 'config': self.config,
            "epoch": cur_epoch,
        }
        torch.save(
            save_obj,
            os.path.join(
                self.output_dir,
                "checkpoint_{}.pth".format("best" if is_best else cur_epoch),
            ),
        )

    def log_stats(self, stats, split_name):
        if isinstance(stats, dict):
            log_stats = {**{f"{split_name}_{k}": v for k, v in stats.items()}}
            with open(os.path.join(self.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")
        elif isinstance(stats, list):
            pass
