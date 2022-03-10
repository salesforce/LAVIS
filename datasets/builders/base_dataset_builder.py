import os

from common.registry import registry

from torchvision.datasets.utils import download_url


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

    def __init__(self, cfg):
        super().__init__()

        self.config = cfg
        self.data_type = cfg.data_type
    
    def build_datasets(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed

        self._download_data()
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.

        datasets = self.build() # dataset['train'/'val'/'test']

        return datasets

    @classmethod
    def default_config_path(cls):
        return None

    def _download_data(self):
        # [TODO] error-handling
        self._download_ann()
        self._download_vis()

    # We need some downloading utilities to help.
    def _download_ann(self):
        """
        Download annotation files if necessary.

        Local annotation paths should be relative. 
        """        
        local_anns = self.config.storage.annotations
        remote_anns = self.config.build_info.annotations

        local_splits = local_anns.keys()
        remote_splits = remote_anns.keys()

        assert local_splits == remote_splits, "Inconsistent remote and local splits, found {} and {}.".format(remote_splits, local_splits)

        cache_root = registry.get_path('cache_root')

        for split in local_splits:
            rel_path = local_anns[split]
            ann_dir, filename = os.path.split(os.path.join(cache_root, rel_path))

            if not os.path.exists(ann_dir): os.makedirs(ann_dir)
            download_url(url=remote_anns[split], root=ann_dir, filename=filename)
        
    # We need some downloading utilities to help.
    def _download_vis(self):
        # downloading images/videos can be dataset-specific.
        raise NotImplementedError

    def build(self):
        # __getitem__() can be dataset-specific.
        raise NotImplementedError
