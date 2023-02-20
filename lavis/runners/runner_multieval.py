"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import datetime
import logging
import time
import torch
import torch.distributed as dist
from lavis.common.dist_utils import is_main_process
from lavis.common.registry import registry
from lavis.datasets.datasets.dataloader_utils import MultiIterLoader
from lavis.runners.runner_iter import RunnerIter


@registry.register_runner("runner_multieval")
class RunnerMultiEval(RunnerIter):
    def __init__(self, cfg, task, model, datasets, job_id):
        super().__init__(cfg, task, model, datasets, job_id)

        self._eval_loaders = None

    def train(self):
        start_time = time.time()
        best_agg_metric = 0
        best_iters = 0

        self.log_config()

        # resume from checkpoint if specified
        if not self.evaluate_only and self.resume_ckpt_path is not None:
            self._load_checkpoint(self.resume_ckpt_path)

        for start_iters in range(
            self.start_iters, self.max_iters, self.iters_per_inner_epoch
        ):
            end_iters = start_iters + self.iters_per_inner_epoch

            # training phase
            if not self.evaluate_only:
                logging.info(
                    "Start training, max_iters={}, in total {} inner epochs.".format(
                        self.max_iters, int(self.max_iters / self.iters_per_inner_epoch)
                    )
                )

                train_stats = self.train_iters(self.cur_epoch, start_iters)
                self.log_stats(split_name="train", stats=train_stats)

            # evaluation phase
            if len(self.task.eval_task_list) > 0:
                eval_log = self.eval_epoch(cur_epoch=self._progress(end_iters))
                self._save_checkpoint(end_iters, is_best=False)

                # TODO (wenliang) add logging results here
            else:
                if not self.evaluate_only:
                    self._save_checkpoint(end_iters, is_best=False)

            if self.evaluate_only:
                break
            dist.barrier()

        # testing phase
        self.evaluate(cur_epoch=self.cur_epoch)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logging.info("Training time {}".format(total_time_str))

    @torch.no_grad()
    def eval_epoch(self, cur_epoch, skip_reload=False):
        """
        Evaluate the model on a given split.

        Args:
            split_name (str): name of the split to evaluate on.
            cur_epoch (int): current epoch.
            skip_reload_best (bool): whether to skip reloading the best checkpoint.
                During training, we will reload the best checkpoint for validation.
                During testing, we will use provided weights and skip reloading the best checkpoint .
        """
        eval_loaders = self.eval_loaders

        # TODO consider moving to model.before_evaluation()
        model = self.unwrap_dist_model(self.model)
        if not skip_reload and cur_epoch == "best":
            model = self._reload_best_model(model)
        model.eval()

        results = self.task.evaluation(cur_epoch, model, eval_loaders)

        if results is not None:
            return self.task.after_evaluation(
                val_result=results,
                epoch=cur_epoch,
            )

    @property
    def eval_loaders(self):
        if self._eval_loaders is None:
            self._eval_loaders = {}
            for split_name in self.dataloaders:
                if split_name != "train":
                    self._eval_loaders[split_name] = self.dataloaders[split_name]

        return self._eval_loaders

    @property
    def dataloaders(self) -> dict:
        """
        A property to get and create dataloaders by split just in need.

        train_dataset_ratio is required, create a MultiIterLoader to sample
        each dataset by ratios during training.

        Validation and test sets are not concatenated.

        Returns:
            dict: {split_name: (tuples of) dataloader}
        """
        if self._dataloaders is None:
            # reoganize datasets by split and concatenate/chain if necessary
            self.datasets = _reorg_named_datasets_by_split(self.datasets)

            # Next, we remove unused datasets
            # remove unused train datasets
            train_datasets_to_keep = {}

            # keep only the datasets that are in the task's train_dataset_ratios
            for name in self.datasets["train"]:
                if name in self.task.train_dataset_names:
                    train_datasets_to_keep[name] = self.datasets["train"][name]
            logging.info(
                "Datasets used for training: {}".format(train_datasets_to_keep.keys())
            )
            self.datasets["train"] = train_datasets_to_keep

            # now we remove unused val and test datasets
            for split in ["val", "test"]:
                new_datasets = {}

                for name in self.datasets[split]:
                    if (name, split) in self.task.eval_dataset_list:
                        new_datasets[name] = self.datasets[split][name]
                    else:
                        logging.info(
                            "Dataset {} is not used for {}".format(name, split)
                        )

                self.datasets[split] = new_datasets

            # up to this point, we have removed unused datasets
            # now we create dataloaders for each split
            train_bsz = self.config.run_cfg.batch_size_train
            eval_bsz = self.config.run_cfg.batch_size_eval

            collate_fns = dict()
            for split in self.datasets:
                collate_fns[split] = dict()
                for name, dataset in self.datasets[split].items():
                    collate_fn = getattr(dataset, "collater", None)

                    collate_fns[split][name] = collate_fn

            num_workers = self.config.run_cfg.num_workers

            # create dataloaders for each split
            loaders = dict()
            for split in self.datasets:
                if split == "train":
                    loader = MultiIterLoader(
                        loaders=[
                            self._create_single_loader(
                                d, num_workers, train_bsz, True, collate_fns[split][n]
                            )
                            for n, d in self.datasets[split].items()
                        ],
                        ratios=self.task.train_dataset_ratios,
                    )
                    loaders[split] = loader
                else:
                    loaders[split] = dict()
                    for name, dataset in self.datasets[split].items():
                        loaders[split][name] = self._create_single_loader(
                            dataset,
                            num_workers,
                            eval_bsz,
                            False,
                            collate_fns[split][name],
                        )

            self._dataloaders = loaders
        return self._dataloaders


def _reorg_named_datasets_by_split(datasets):
    """
    Organizes datasets by split.

    Args:
        datasets: dict of torch.utils.data.Dataset objects by name.

    Returns:
        Dict of datasets by split {split_name: List[Datasets]}.
    """
    # if len(datasets) == 1:
    #     return datasets[list(datasets.keys())[0]]
    # else:
    reorg_datasets = dict()

    # reorganize by split
    for dataset_name, dataset in datasets.items():
        for split_name, dataset_split in dataset.items():
            if split_name not in reorg_datasets:
                reorg_datasets[split_name] = dict()
                reorg_datasets[split_name][dataset_name] = dataset_split
            else:
                reorg_datasets[split_name][dataset_name] = dataset_split

    return reorg_datasets
