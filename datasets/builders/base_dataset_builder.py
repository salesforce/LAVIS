import logging
import os
import shutil

import torch.distributed as dist
import utils.blip_utils as utils

from common.registry import registry

from torchvision.datasets.utils import download_url


class BaseDatasetBuilder:
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

        if utils.is_main_process():
            self._download_data()

        if utils.is_dist_avail_and_initialized():
            dist.barrier()

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
        vis_train_cfg = self.config.vis_processor.get('train', None)
        vis_eval_cfg = self.config.vis_processor.get('eval', None)

        text_train_cfg = self.config.text_processor.get('train', None)
        text_eval_cfg = self.config.text_processor.get('eval', None)

        self.vis_processors['train'] = self._build_from_config(vis_train_cfg)
        self.vis_processors['eval'] = self._build_from_config(vis_eval_cfg)

        self.text_processors['train'] = self._build_from_config(text_train_cfg)
        self.text_processors['eval'] = self._build_from_config(text_eval_cfg)
        
    @staticmethod
    def _build_from_config(cfg):
        if cfg is None:
            return None
        else:
            return registry.get_processor_class(cfg.name).build_from_cfg(cfg)

    @staticmethod
    def save_build_info(build_info_path, url):
        from datetime import datetime
        import json

        info = {"date": str(datetime.now()), "from": url}
        json.dump(info, open(build_info_path, 'w+'))

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
        anns = self.config.build_info.annotations

        splits = anns.keys()

        cache_root = registry.get_path('cache_root')

        for split in splits:
            info = anns[split]

            urls, storage_paths = info.url, info.storage

            if isinstance(urls, str): urls = [urls]
            if isinstance(storage_paths, str): storage_paths = [storage_paths]

            assert len(urls) == len(storage_paths)

            for url_or_filename, storage_path in zip(urls, storage_paths):
                # if storage_path is relative, make it full by prefixing with cache_root.
                if not os.path.isabs(storage_path):
                    storage_path = os.path.join(cache_root, storage_path)

                if os.path.isfile(url_or_filename):
                    src, dst = url_or_filename, storage_path
                    if not os.path.exists(dst):
                        shutil.copyfile(src=src, dst=dst)
                    else:
                        logging.info('Using existing file {}.'.format(url_or_filename))
                else:
                    dirname = os.path.dirname(storage_path)
                    if not os.path.exists(dirname): os.makedirs(dirname)

                    if os.path.isdir(storage_path):
                        # if only dirname is provided, suffix with basename of URL.
                        raise ValueError('Expecting storage_path to be a file path, got directory {}'.format(storage_path))
                    else:
                        filename = os.path.basename(storage_path)

                    # download_url(url=url, root=dirname, filename=filename, md5=info.md5)
                    download_url(url=url_or_filename, root=dirname, filename=filename)
        
    # We need some downloading utilities to help.
    def _download_vis(self):
        # downloading images/videos can be dataset-specific.
        raise NotImplementedError

    def build(self):
        """
        Create by split datasets inheriting torch.utils.data.Datasets.

        # build() can be dataset-specific. Overwrite to customize.
        """
        self.build_processors()

        build_info = self.config.build_info
        
        ann_info = build_info.annotations
        vis_info = build_info.get(self.data_type)

        datasets = dict()
        for split in ann_info.keys():
            assert split in ['train', 'val', 'test'], "Invalid split name {}, must be one of 'train', 'val' and 'test'."
            is_train = split == 'train'

            # processors
            vis_processor = self.vis_processors['train'] if is_train else self.vis_processors['eval']
            text_processor = self.text_processors['train'] if is_train else self.text_processors['eval']

            # annotation path
            ann_paths = ann_info.get(split).storage
            if isinstance(ann_paths, str): ann_paths = [ann_paths]

            abs_ann_paths = []
            for ann_path in ann_paths:
                if not os.path.isabs(ann_path):
                    ann_path = os.path.join(registry.get_path("cache_root"), ann_path) 
                abs_ann_paths.append(ann_path)
            ann_paths = abs_ann_paths

            # visual data paths
            vis_paths = vis_info.get(split).storage
            if isinstance(vis_paths, str): vis_paths = [vis_paths]

            abs_vis_paths = []
            for vis_path in vis_paths:
                if not os.path.isabs(vis_path):
                    vis_path = os.path.join(registry.get_path("cache_root"), vis_path) 
                abs_vis_paths.append(vis_path)
            vis_paths = abs_vis_paths

            # create datasets
            dataset_cls = self.train_dataset_cls if is_train else self.eval_dataset_cls
            datasets[split] = dataset_cls(
                    vis_processor=vis_processor,
                    text_processor=text_processor,
                    ann_paths=ann_paths,
                    image_roots=vis_paths
                )

        return datasets