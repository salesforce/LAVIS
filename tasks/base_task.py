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
        """
        Do forward and backward, and return the loss as computed by *criterion*
        for the given *model* and *sample*.

        Args:
            sample (dict): the mini-batch.
            model (models.BaseModel): the model

        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """
        loss = model(samples)["loss"]
        return loss

    def valid_step(self, model, samples):
        raise NotImplementedError

    def after_evaluation(self, **kwargs):
        pass

    def inference_step(self):
        raise NotImplementedError

    def optimizer_step(self, optimizer, model, update_num):
        optimizer.step()
