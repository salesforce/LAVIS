import logging
import os

import torch.distributed as dist
from lavis.common.dist_utils import get_rank, get_world_size, is_main_process
from lavis.common.logger import MetricLogger, SmoothedValue
from lavis.common.registry import registry
from lavis.datasets.data_utils import concat_datasets, prepare_sample


class BaseTask:
    def __init__(self, **kwargs):
        super().__init__()

    @classmethod
    def setup_task(cls, **kwargs):
        return cls()

    def build_model(self, cfg):
        model_config = cfg.model_cfg

        model_cls = registry.get_model_class(model_config.arch)
        return model_cls.build(model_config)

    def build_datasets(self, cfg):
        """
        Build a dictionary of datasets, keyed by split 'train', 'valid', 'test'.
        Download dataset and annotations automatically if not exist.

        Args:
            cfg (common.config.Config): _description_

        Returns:
            multi_datasets (List): _description_
        """

        multi_datasets = dict()

        datasets_config = cfg.datasets_cfg

        assert len(datasets_config) > 0, "At least one dataset has to be specified."

        for name in datasets_config:
            dataset_config = datasets_config[name]

            builder = registry.get_builder_class(name)(dataset_config)
            datasets = builder.build_datasets()

            multi_datasets[name] = datasets

        datasets = concat_datasets(multi_datasets)
        for split_name in datasets:
            if hasattr(datasets[split_name], "__len__"):
                logging.info(
                    "Loaded {} records for {} split.".format(
                        len(datasets[split_name]), split_name
                    )
                )
        return datasets

    def train_step(self, model, samples):
        loss = model(samples)["loss"]
        return loss

    def valid_step(self, model, samples):
        raise NotImplementedError

    def after_evaluation(self, **kwargs):
        pass

    def inference_step(self):
        raise NotImplementedError

    def create_eval_model(self, model, **kwargs):
        return model

    def evaluation(self, model, data_loader, cuda_enabled=True):
        metric_logger = MetricLogger(delimiter="  ")
        header = "Evaluation"
        # TODO make it configurable
        print_freq = 10

        results = []

        for samples in metric_logger.log_every(data_loader, print_freq, header):
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)

            eval_output = self.valid_step(model=model, samples=samples)
            results.extend(eval_output)

        dist.barrier()

        return results

    def train_epoch(
        self,
        epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        cuda_enabled=True,
        log_freq=50,
    ):
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))

        header = "Train Epoch: [{}]".format(epoch)

        for i, samples in enumerate(
            metric_logger.log_every(data_loader, log_freq, header)
        ):
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            samples.update(
                {"epoch": epoch, "num_iters_per_epoch": len(data_loader), "iters": i}
            )

            lr_scheduler.step(cur_epoch=epoch, cur_step=i)
            optimizer.zero_grad()

            loss = self.train_step(model=model, samples=samples)

            # after_train_step()
            loss.backward()
            optimizer.step()

            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # after train_epoch()
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        logging.info("Averaged stats: " + str(metric_logger.global_avg()))
        return {
            k: "{:.3f}".format(meter.global_avg)
            for k, meter in metric_logger.meters.items()
        }

    @staticmethod
    def save_result(result, result_dir, filename, remove_duplicate=""):
        import json

        logging.warning("rank %d starts dumping." % get_rank())
        result_file = os.path.join(
            result_dir, "%s_rank%d.json" % (filename, get_rank())
        )
        logging.warning("rank %d finish dumping." % get_rank())
        final_result_file = os.path.join(result_dir, "%s.json" % filename)

        json.dump(result, open(result_file, "w"))

        dist.barrier()

        if is_main_process():
            logging.warning("rank %d starts merging." % get_rank())
            # combine results from all processes
            result = []

            for rank in range(get_world_size()):
                result_file = os.path.join(
                    result_dir, "%s_rank%d.json" % (filename, rank)
                )
                res = json.load(open(result_file, "r"))
                result += res

            if remove_duplicate:
                result_new = []
                id_list = []
                for res in result:
                    if res[remove_duplicate] not in id_list:
                        id_list.append(res[remove_duplicate])
                        result_new.append(res)
                result = result_new

            json.dump(result, open(final_result_file, "w"))
            print("result file saved to %s" % final_result_file)

        return final_result_file
