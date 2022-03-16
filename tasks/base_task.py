import torch

from common.registry import registry

class BaseTask:
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        # Dict: split (str) -> dataset (Dataset)
        self.datasets = dict()

    @classmethod
    def setup_task(cls, cfg, **kwargs):
        return cls(cfg, **kwargs)
    
    def load_dataset(self, split):
        raise NotImplementedError

    def dataset(self, split):
        """Return a dataset split."""
        if split not in self.datasets:
            raise KeyError('Dataset not loaded: ' + split)
        # if not isinstance(self.datasets[split], FairseqDataset):
        #     raise TypeError('Datasets are expected to be of type FairseqDataset')
        return self.datasets[split]

    def build_model(self, cfg):
        raise NotImplementedError

    def build_criterion(self, cfg):
        raise NotImplementedError

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

        datasets_config = cfg.get_datasets_config()

        # [TODO] to support multiple datasets
        assert len(datasets_config) > 0, "At least one dataset has to be specified."
        assert len(datasets_config) == 1, "Do not support multiple datasets for now."

        for name in datasets_config:
            dataset_config = datasets_config[name]

            builder = registry.get_builder_class(name)(dataset_config)
            datasets = builder.build_datasets()

            multi_datasets[name] = datasets
        
        return multi_datasets

    def train_step(self, sample, model, criterion, optimizer):
        """
        Do forward and backward, and return the loss as computed by *criterion*
        for the given *model* and *sample*.

        Args:
            sample (dict): the mini-batch.
            model (models.BaseModel): the model
            criterion (Criterion): the criterion
            optimizer (Optimizer): the optimizer
            update_num (int): the current update
            ignore_grad (bool): multiply loss by 0 if this is set to True

        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """ 
        model.train()
        pass

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            loss, sample_size, logging_output = criterion(model, sample)
        return loss, sample_size, logging_output
    
    def inference_step(self):
        raise NotImplementedError

    def optimizer_step(self, optimizer, model, update_num):
        optimizer.step() 