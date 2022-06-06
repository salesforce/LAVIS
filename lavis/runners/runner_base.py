import datetime
import logging
import os
import json
import time
from pathlib import Path

import torch
import torch.distributed as dist

import lavis.common.utils as utils
from lavis.common.registry import registry
from torch.utils.data import DataLoader


class Runner:
    def __init__(self, cfg, task, model, datasets, job_id):
        self.config = cfg
        self.job_id = job_id

        self.task = task
        self.datasets = datasets

        self._model = model

        self._wrapped_model = None
        self._device = None
        self._optimizer = None
        self._dataloaders = None
        self._lr_sched = None

        # self.setup_seeds()
        self.setup_output_dir()

    @property
    def device(self):
        if self._device is None:
            self._device = torch.device(self.config.run_cfg.device)

        return self._device

    @property
    def use_distributed(self):
        return self.config.run_cfg.distributed

    @property
    def model(self):
        if self._model.device != self.device:
            self._model = self._model.to(self.device)

            if self.use_distributed:
                if self._wrapped_model is None:
                    self._wrapped_model = torch.nn.parallel.DistributedDataParallel(
                        self._model, device_ids=[self.config.run_cfg.gpu]
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
                lr=float(self.config.run_cfg.init_lr),
                weight_decay=float(self.config.run_cfg.weight_decay),
            )

        return self._optimizer

    @property
    def lr_scheduler(self):
        if self._lr_sched is None:
            lr_sched_cls = registry.get_lr_scheduler_class(self.config.run_cfg.lr_sched)

            max_epoch = self.config.run_cfg.max_epoch
            min_lr = self.config.run_cfg.min_lr
            init_lr = self.config.run_cfg.init_lr

            # optional parameters
            decay_rate = self.config.run_cfg.get("lr_decay_rate", None)
            warmup_start_lr = self.config.run_cfg.get("warmup_lr", -1)
            warmup_steps = self.config.run_cfg.get("warmup_steps", 0)

            self._lr_sched = lr_sched_cls(
                optimizer=self.optimizer,
                max_epoch=max_epoch,
                min_lr=min_lr,
                init_lr=init_lr,
                decay_rate=decay_rate,
                warmup_start_lr=warmup_start_lr,
                warmup_steps=warmup_steps,
            )

        return self._lr_sched

    @property
    def dataloaders(self):
        if self._dataloaders is None:

            split_names = sorted(self.datasets.keys())

            datasets = [self.datasets[split] for split in split_names]
            is_train = [split in self.train_splits for split in split_names]

            if self.use_distributed:
                samplers = create_sampler(
                    datasets=datasets,
                    shuffles=is_train,
                    num_tasks=utils.get_world_size(),
                    global_rank=utils.get_rank(),
                )
                if not self.use_dist_eval_sampler:
                    # e.g. retrieval evaluation
                    dist_samplers = [
                        sampler if flag else None
                        for sampler, flag in zip(samplers, is_train)
                    ]
                    samplers = dist_samplers
            else:
                samplers = [None] * len(self.datasets)

            dataloaders = create_loader(
                datasets=datasets,
                samplers=samplers,
                batch_size=[
                    self.config.run_cfg.batch_size_train
                    if split == "train"
                    else self.config.run_cfg.batch_size_eval
                    for split in split_names
                ],
                num_workers=[self.config.run_cfg.num_workers] * len(datasets),
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
        return int(self.config.run_cfg.max_epoch)

    @property
    def log_freq(self):
        log_freq = self.config.run_cfg.get("log_freq", 50)
        return int(log_freq)

    @property
    def init_lr(self):
        return float(self.config.run_cfg.init_lr)

    @property
    def min_lr(self):
        return float(self.config.run_cfg.min_lr)

    @property
    def valid_splits(self):
        valid_splits = self.config.run_cfg.get("valid_splits", [])

        if len(valid_splits) == 0:
            logging.warning("No validation splits found.")

        return valid_splits

    @property
    def test_splits(self):
        test_splits = self.config.run_cfg.get("test_splits", [])

        return test_splits

    @property
    def train_splits(self):
        train_splits = self.config.run_cfg.get("train_splits", [])

        if len(train_splits) == 0:
            logging.info("Empty train splits.")

        return train_splits

    @property
    def evaluate_only(self):
        return self.config.run_cfg.evaluate

    @property
    def use_dist_eval_sampler(self):
        return self.config.run_cfg.get("use_dist_eval_sampler", True)

    @property
    def train_loader(self):
        train_loader = self.dataloaders["train"]

        assert isinstance(train_loader, DataLoader)
        return train_loader

    def setup_output_dir(self):
        lib_root = Path(registry.get_path("library_root"))

        output_dir = lib_root / self.config.run_cfg.output_dir / self.job_id
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
        best_epoch = 0

        for cur_epoch in range(0, self.max_epoch):
            # training phase
            if not self.evaluate_only:
                logging.info("Start training")
                if self.use_distributed:
                    self.train_loader.sampler.set_epoch(cur_epoch)

                train_stats = self.train_epoch(cur_epoch)

                if utils.is_main_process():
                    self.log_stats(split_name="train", stats=train_stats)

            # evaluation phase
            if len(self.valid_splits) > 0:
                for split_name in self.valid_splits:
                    logging.info("Evaluating on {}.".format(split_name))

                    val_result = self.evaluation(split_name=split_name)
                    if val_result is None:
                        continue

                    val_log = self.task.after_evaluation(
                        val_result=val_result, split_name=split_name, epoch=cur_epoch
                    )

                    if utils.is_main_process():
                        assert (
                            "agg_metrics" in val_log
                        ), "agg_metrics must be present in evaluation log if validation set is used."

                        agg_metrics = val_log["agg_metrics"]
                        if agg_metrics > best_agg_metric and split_name == "val":
                            best_epoch = cur_epoch
                            best_agg_metric = agg_metrics

                            self.save_checkpoint(cur_epoch, is_best=True)

                        val_log.update({"best_epoch": best_epoch})
                        self.log_stats(val_log, split_name)

            else:
                # no validation split is provided.
                if not self.evaluate_only and utils.is_main_process():
                    self.save_checkpoint(cur_epoch, is_best=False)

            if self.evaluate_only:
                break
            dist.barrier()

        # testing phase
        self.evaluate(cur_epoch=cur_epoch)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logging.info("Training time {}".format(total_time_str))

    def evaluate(self, cur_epoch="best"):
        test_logs = dict()

        if len(self.test_splits) > 0:
            for split_name in self.test_splits:
                test_result = self.evaluation(split_name=split_name)
                test_log = self.task.after_evaluation(
                    val_result=test_result,
                    split_name=split_name,
                    epoch=cur_epoch,
                    result_dir=self.result_dir,
                )
                test_logs[split_name] = test_log

            return test_logs

    def train_epoch(self, epoch):
        # train
        self.model.train()

        return self.task.train_epoch(
            epoch=epoch,
            model=self.model,
            data_loader=self.train_loader,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            cuda_enabled=self.cuda_enabled,
            log_freq=self.log_freq,
        )

    @torch.no_grad()
    def evaluation(self, split_name):
        # TODO In validation, you need to compute loss as well as metrics
        model = self.model_without_ddp
        model.eval()

        data_loader = self.dataloaders.get(split_name, None)
        assert data_loader, "data_loader for split {} is None.".format(split_name)

        results = self.task.evaluation(model, data_loader)

        return results

    def save_checkpoint(self, cur_epoch, is_best=False):
        save_obj = {
            "model": self.model_without_ddp.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.config.to_dict(),
            "epoch": cur_epoch,
        }
        save_to = os.path.join(
            self.output_dir,
            "checkpoint_{}.pth".format("best" if is_best else cur_epoch),
        )
        logging.info("Saving checkpoint at epoch {} to {}.".format(cur_epoch, save_to))
        torch.save(save_obj, save_to)

    def log_stats(self, stats, split_name):
        if isinstance(stats, dict):
            log_stats = {**{f"{split_name}_{k}": v for k, v in stats.items()}}
            with open(os.path.join(self.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")
        elif isinstance(stats, list):
            pass


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset, sampler, bs, n_worker, is_train, collate_fn in zip(
        datasets, samplers, batch_size, num_workers, is_trains, collate_fns
    ):
        if is_train:
            shuffle = sampler is None
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )
        loaders.append(loader)
    return loaders


def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset, shuffle in zip(datasets, shuffles):
        sampler = torch.utils.data.DistributedSampler(
            dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle
        )
        samplers.append(sampler)
    return samplers
