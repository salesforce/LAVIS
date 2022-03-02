class BaseDatasetBuilder:
    """A BaseDatasetBuilder standardizes the training, val, test splits, data preparation and transforms. The main
    advantage is consistent data splits, data preparation and transforms across models.
    Example::
        class MyDatasetBuilder(BaseDatasetBuilder):
            def __init__(self):
                super().__init__()
            def prepare_data(self):
                # download, split, etc...
                # only called on 1 GPU/TPU in distributed
            def setup(self, stage):
                # make assignments here (val/train/test split)
                # called on every process in DDP
            def train_dataloader(self):
                train_split = Dataset(...)
                return DataLoader(train_split)
            def val_dataloader(self):
                val_split = Dataset(...)
                return DataLoader(val_split)
            def test_dataloader(self):
                test_split = Dataset(...)
                return DataLoader(test_split)
    """

    def __init__(self):
        super().__init__()
    
    def prepare_data(self, train_transforms=None, val_transforms=None, test_transforms=None):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
        pass

    @classmethod
    def default_config_path(cls):
        return None