from lavis.common.registry import Registry, registry
from lavis.tasks.base_task import BaseTask


@Registry.register_task("instruct_tuning")
class InstructTuningTask(BaseTask):
    """
    This class is used to do multitask training and evaluation.

    TODO (dxli) support multiple training tasks.
    - One training task can be specified;
    - Multiple evaluation tasks can be specified;
    - for each task, multiple datasets can be specified;
    - for evaluation tasks, each task will be evaluated in order.
    """

    def __init__(self, train_dataset_name2ratio, eval_task_list, eval_dataset_list):
        self.train_dataset_name2ratio = train_dataset_name2ratio

        self.eval_task_list = eval_task_list
        self.eval_dataset_list = eval_dataset_list

        assert len(eval_task_list) == len(
            eval_dataset_list
        ), "Number of evaluation tasks and datasets should be the same."

    @property
    def train_dataset_names(self):
        return list(self.train_dataset_name2ratio.keys())

    @property
    def train_dataset_ratios(self):
        return list(self.train_dataset_name2ratio.values())

    def evaluation(self, model, dataloaders, cuda_enabled=True):
        """
        dataloaders: a dict of {split: {dataset_name: dataloader}}

        """
        task_split_dataloader = []

        # TODO one corner case is that the same dataset is used for multiple tasks.
        # however, this should in general not happening. If one dataset is repurposed
        # for multiple tasks, it should be split into multiple datasets.
        for task, dataset in zip(self.eval_task_list, self.eval_dataset_list):
            dataset_name, split = dataset
            task_split_dataloader.append(
                (task, split, dataloaders[split][dataset_name])
            )

        all_val_results = []

        for task, split, dataloader in task_split_dataloader:
            val_results = task.evaluation(model, dataloader, cuda_enabled=cuda_enabled)
            # TODO (wenliang) please properly name the split_name
            metrics = task.after_evaluation(val_results, split_name=split)
            all_val_results.append(metrics)

        print("all_val_results: ", all_val_results)
        return all_val_results

    def build_datasets(self, cfg):
        datasets = super().build_datasets(cfg)

        # a little hacky to get question and answer list for vqa task, see vqa.after_build_datasets().
        for task, dataset in zip(self.eval_task_list, self.eval_dataset_list):
            dataset_name, _ = dataset

            datasets[dataset_name] = task.after_build_datasets(
                {dataset_name: datasets[dataset_name]}
            )[dataset_name]

        return datasets

    @classmethod
    def setup_task(cls, cfg):
        task_config = cfg.run_cfg.task_config

        eval_task_list, eval_dataset_list = [], []

        task_config_train = task_config.train
        task_config_eval = task_config.eval

        # training related
        train_dataset_name2ratio = task_config_train.get("dataset_name2ratio", None)
        if train_dataset_name2ratio is None:
            raise ValueError(
                "Dataset ratios must be specified for {} task.".format(__class__)
            )

        # evaluation task
        for item in task_config_eval:
            task_name = list(item.keys())[0]

            # setup task
            # if no args, default task arguments will be used.
            task_args = item[task_name].get("args", dict())
            task = registry.get_task_class(task_name).setup_task(cfg=task_args)
            assert task is not None, "Task {} not properly registered.".format(
                task_name
            )
            eval_task_list.append(task)

            assert (
                "dataset" in item[task_name]
            ), "dataset to use must be specified for task {}.".format(task_name)
            task_datasets = item[task_name].dataset
            split = item[task_name].split
            eval_dataset_list.append((task_datasets, split))

        return cls(
            train_dataset_name2ratio=train_dataset_name2ratio,
            eval_task_list=eval_task_list,
            eval_dataset_list=eval_dataset_list,
        )
