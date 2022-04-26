import logging
import torch.distributed as dist
from datasets.data_utils import prepare_sample
import utils.blip_utils as utils

from common.registry import registry
from utils.data_utils import concat_datasets


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
            cfg (utils.config.Config): _description_

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

        return concat_datasets(multi_datasets)

    def train_step(self, model, samples):
        loss = model(samples)["loss"]
        return loss

    def valid_step(self, model, samples):
        raise NotImplementedError

    def after_evaluation(self, **kwargs):
        pass

    def inference_step(self):
        raise NotImplementedError

    # def optimizer_step(self, optimizer, model, update_num):
    # # this could be useful for accumulating gradients
    #     optimizer.step()

    def evaluation(self, model, data_loader, cuda_enabled=True):
        metric_logger = utils.MetricLogger(delimiter="  ")
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
    
    def train_epoch(self, epoch, model, data_loader, optimizer, cuda_enabled=True, log_freq=50):
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter("loss", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))

        header = "Train Epoch: [{}]".format(epoch)

        for i, samples in enumerate(metric_logger.log_every(data_loader, log_freq, header)):
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            samples.update(
                {
                    "epoch": epoch,
                    "num_iters_per_epoch": len(data_loader),
                    "iters": i
                }
            )

            loss = self.train_step(model=model, samples=samples)

            # after_train_step()
            optimizer.zero_grad()
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