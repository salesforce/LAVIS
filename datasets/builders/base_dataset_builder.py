import os
from tkinter import E

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
        self._download_ann()
        self._download_vis()

    def _download_ann(self):
        """
        Download annotation files if necessary. 
        All the vision-language datasets should have annotations of unified format.

        storage_path can be:
          (1) relative/absolute: will be prefixed with env.cache_root to make full path if relative.
          (2) basename/dirname: will be suffixed with base name of URL if dirname is provided.

        Local annotation paths should be relative. 
        """        
        local_anns = self.config.storage.annotations
        remote_anns = self.config.build_info.annotations

        local_splits = local_anns.keys()
        remote_splits = remote_anns.keys()

        assert local_splits == remote_splits, "Inconsistent remote and local splits, found {} and {}.".format(remote_splits, local_splits)

        cache_root = registry.get_path('cache_root')

        for split in local_splits:
            storage_path = local_anns[split]
            remote_info = remote_anns[split]

            # if storage_path is relative, make it full by prefixing with cache_root.
            if not os.path.isabs(storage_path):
                storage_path = os.path.join(cache_root, storage_path)

            # create the directory if not exist
            dirname = os.path.dirname(storage_path)
            if not os.path.exists(dirname): os.makedirs(dirname)

            if os.path.isdir(storage_path):
                # if only dirname is provided, suffix with basename of URL.
                raise ValueError('Expecting storage_path to be a file path, got directory {}'.format(storage_path))
            else:
                filename = os.path.basename(storage_path)

            download_url(url=remote_info.url, root=dirname, filename=filename, md5=remote_info.md5)
        
    # We need some downloading utilities to help.
    def _download_vis(self):
        # downloading images/videos can be dataset-specific.
        raise NotImplementedError

    def build(self):
        # __getitem__() can be dataset-specific.
        raise NotImplementedError
