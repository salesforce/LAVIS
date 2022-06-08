import logging
import os
import shutil

import lavis.common.utils as utils
import torch.distributed as dist

from lavis.common.dist_utils import is_dist_avail_and_initialized, is_main_process
from lavis.common.registry import registry
from lavis.datasets.data_utils import extract_archive
from lavis.processors.base_processor import BaseProcessor
from omegaconf import OmegaConf
from torchvision.datasets.utils import download_url


class BaseDatasetBuilder:
    train_dataset_cls, eval_dataset_cls = None, None

    def __init__(self, cfg=None):
        super().__init__()

        if cfg is None:
            # help to create datasets from default config.
            cfg = OmegaConf.load(self.default_config_path()).datasets
            self.config = cfg[list(cfg.keys())[0]]
        elif isinstance(cfg, str):
            cfg = OmegaConf.load(cfg).datasets
            self.config = cfg[list(cfg.keys())[0]]
        else:
            self.config = cfg

        self.data_type = self.config.data_type

        self.vis_processors = {"train": BaseProcessor(), "eval": BaseProcessor()}
        self.text_processors = {"train": BaseProcessor(), "eval": BaseProcessor()}

    def build_datasets(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed

        if is_main_process():
            self._download_data()

        if is_dist_avail_and_initialized():
            dist.barrier()

        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        datasets = self.build()  # dataset['train'/'val'/'test']

        return datasets

    def build_processors(self):
        vis_proc_cfg = self.config.get("vis_processor")
        txt_proc_cfg = self.config.get("text_processor")

        if vis_proc_cfg is not None:
            vis_train_cfg = vis_proc_cfg.get("train")
            vis_eval_cfg = vis_proc_cfg.get("eval")

            self.vis_processors["train"] = self._build_proc_from_cfg(vis_train_cfg)
            self.vis_processors["eval"] = self._build_proc_from_cfg(vis_eval_cfg)

        if txt_proc_cfg is not None:
            txt_train_cfg = txt_proc_cfg.get("train")
            txt_eval_cfg = txt_proc_cfg.get("eval")

            self.text_processors["train"] = self._build_proc_from_cfg(txt_train_cfg)
            self.text_processors["eval"] = self._build_proc_from_cfg(txt_eval_cfg)

    @staticmethod
    def _build_proc_from_cfg(cfg):
        return (
            registry.get_processor_class(cfg.name).build_from_cfg(cfg)
            if cfg is not None
            else None
        )

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

        cache_root = registry.get_path("cache_root")

        for split in splits:
            info = anns[split]

            urls, storage_paths = info.get("url", None), info.storage

            if isinstance(urls, str):
                urls = [urls]
            if isinstance(storage_paths, str):
                storage_paths = [storage_paths]

            assert len(urls) == len(storage_paths)

            for url_or_filename, storage_path in zip(urls, storage_paths):
                # if storage_path is relative, make it full by prefixing with cache_root.
                if not os.path.isabs(storage_path):
                    storage_path = os.path.join(cache_root, storage_path)

                dirname = os.path.dirname(storage_path)
                if not os.path.exists(dirname):
                    os.makedirs(dirname)

                if os.path.isfile(url_or_filename):
                    src, dst = url_or_filename, storage_path
                    if not os.path.exists(dst):
                        shutil.copyfile(src=src, dst=dst)
                    else:
                        logging.info("Using existing file {}.".format(dst))
                else:
                    if os.path.isdir(storage_path):
                        # if only dirname is provided, suffix with basename of URL.
                        raise ValueError(
                            "Expecting storage_path to be a file path, got directory {}".format(
                                storage_path
                            )
                        )
                    else:
                        filename = os.path.basename(storage_path)

                    # download_url(url=url, root=dirname, filename=filename, md5=info.md5)
                    download_url(url=url_or_filename, root=dirname, filename=filename)

    def _download_vis(self):

        data_type = self.data_type
        vis_urls = self.vis_urls[data_type]

        cache_root = registry.get_path("cache_root")

        # Create temp directory for caching downloads.
        dl_cache_dir = os.path.join(cache_root, "temp/coco")
        os.makedirs(dl_cache_dir, exist_ok=True)

        storage_path = self.config.build_info.get(self.data_type).storage

        new_build = False

        for split in vis_urls.keys():
            if not os.path.isabs(storage_path):
                storage_path = os.path.join(cache_root, storage_path)

            if os.path.exists(storage_path) and not new_build:
                logging.info(
                    "Data build path {} exists, skip downloading.".format(storage_path)
                )
            else:  # if path not exist or build for the first time
                new_build = True

                urls = vis_urls[split]

                if isinstance(urls, str):
                    urls = [urls]

                for url in urls:
                    # note this skips the downloading if the file already exists
                    download_url(url=url, root=dl_cache_dir)

                    dirname = os.path.dirname(storage_path)
                    assert os.path.normpath(dirname) == os.path.normpath(
                        storage_path
                    ), "Local path to store images has to be a directory, found {}.".format(
                        storage_path
                    )

                    if not os.path.exists(dirname):
                        os.makedirs(dirname)

                    # extracting
                    archive_path = os.path.join(dl_cache_dir, os.path.basename(url))
                    extract_archive(
                        from_path=archive_path, to_path=storage_path, overwrite=False
                    )

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
            assert split in [
                "train",
                "val",
                "test",
            ], "Invalid split name {}, must be one of 'train', 'val' and 'test'."
            is_train = split == "train"

            # processors
            vis_processor = (
                self.vis_processors["train"]
                if is_train
                else self.vis_processors["eval"]
            )
            text_processor = (
                self.text_processors["train"]
                if is_train
                else self.text_processors["eval"]
            )

            # annotation path
            ann_paths = ann_info.get(split).storage
            if isinstance(ann_paths, str):
                ann_paths = [ann_paths]

            abs_ann_paths = []
            for ann_path in ann_paths:
                if not os.path.isabs(ann_path):
                    ann_path = os.path.join(registry.get_path("cache_root"), ann_path)
                abs_ann_paths.append(ann_path)
            ann_paths = abs_ann_paths

            # visual data storage path
            vis_path = vis_info.storage

            if not os.path.isabs(vis_path):
                vis_path = os.path.join(registry.get_path("cache_root"), vis_path)

            # create datasets
            dataset_cls = self.train_dataset_cls if is_train else self.eval_dataset_cls
            datasets[split] = dataset_cls(
                vis_processor=vis_processor,
                text_processor=text_processor,
                ann_paths=ann_paths,
                image_root=vis_path,
            )

        return datasets
