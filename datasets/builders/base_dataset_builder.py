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
    train_dataset_cls, eval_dataset_cls = None, None

    def __init__(self, cfg):
        super().__init__()

        self.config = cfg
        self.data_type = cfg.data_type

        self.vis_processors = dict()
        self.text_processors = dict()
    
    def build_datasets(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed

        self._download_data()
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.

        datasets = self.build() # dataset['train'/'val'/'test']

        return datasets

    def build_processors(self):
        """
        Building processors is task-specific.

        Some tasks require both visual and text processors, e.g. retrieval, QA.
        Some tasks require only visual processors, e.g. captioning.

        Overwrite for data-specific processors.
        """
        self.vis_processors['train'] = registry.get_processor_class(self.config.vis_processor['train'])()
        self.vis_processors['eval'] = registry.get_processor_class(self.config.vis_processor['eval'])()

        self.text_processors['train'] = registry.get_processor_class(self.config.text_processor['train'])()
        self.text_processors['eval'] = registry.get_processor_class(self.config.text_processor['eval'])()

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
        raise NotImplementedError

    def build(self):
        """
        Create by split datasets inheriting torch.utils.data.Datasets.

        # build() can be dataset-specific. Overwrite to customize.
        """
        self.build_processors()

        storage_info = self.config.storage
        
        ann_paths = storage_info.get('annotations')
        vis_paths = storage_info.get(self.data_type)

        datasets = dict()
        for split in ann_paths.keys():
            assert split in ['train', 'val', 'test'], "Invalid split name {}, must be one of 'train', 'val' and 'test'."
            is_train = split == 'train'

            # processors
            vis_processor = self.vis_processors['train'] if is_train else self.vis_processors['eval']
            text_processor = self.text_processors['train'] if is_train else self.text_processors['eval']

            # annotation path
            ann_path = ann_paths.get(split)
            if not os.path.isabs(ann_path):
                ann_path = os.path.join(registry.get_path("cache_root"), ann_path) 
            
            vis_path = vis_paths.get(split)
            if not os.path.isabs(vis_path):
                vis_path = os.path.join(registry.get_path("cache_root"), vis_path) 

            # create datasets
            dataset_cls = self.train_dataset_cls if is_train else self.eval_dataset_cls
            datasets[split] = dataset_cls(
                    vis_processor=vis_processor,
                    text_processor=text_processor,
                    ann_path=ann_path,
                    image_root=vis_path
                )

        return datasets