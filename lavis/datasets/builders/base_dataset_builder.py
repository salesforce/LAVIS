"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
import os
import shutil
import warnings

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
            self.config = load_dataset_config(self.default_config_path())
        elif isinstance(cfg, str):
            self.config = load_dataset_config(cfg)
        else:
            # when called from task.build_dataset()
            self.config = cfg

        self.data_type = self.config.data_type

        self.vis_processors = {"train": BaseProcessor(), "eval": BaseProcessor()}
        self.text_processors = {"train": BaseProcessor(), "eval": BaseProcessor()}

        # additional processors, each specified by a name in string.
        self.kw_processors = {}

    def build_datasets(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed

        if is_main_process():
            self._download_data()

        if is_dist_avail_and_initialized():
            dist.barrier()

        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
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
        
        kw_proc_cfg = self.config.get("kw_processor")
        if kw_proc_cfg is not None:
            for name, cfg in kw_proc_cfg.items():
                self.kw_processors[name] = self._build_proc_from_cfg(cfg)
        
    @staticmethod
    def _build_proc_from_cfg(cfg):
        return (
            registry.get_processor_class(cfg.name).from_config(cfg)
            if cfg is not None
            else None
        )

    @classmethod
    def default_config_path(cls, type="default"):
        return utils.get_abs_path(cls.DATASET_CONFIG_DICT[type])

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

                    download_url(url=url_or_filename, root=dirname, filename=filename)

    def _download_vis(self):

        storage_path = self.config.build_info.get(self.data_type).storage
        storage_path = utils.get_cache_path(storage_path)

        if not os.path.exists(storage_path):
            warnings.warn(
                f"""
                The specified path {storage_path} for visual inputs does not exist.
                Please provide a correct path to the visual inputs or
                refer to datasets/download_scripts/README.md for downloading instructions.
                """
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
            if split not in ["train", "val", "test"]:
                continue

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
                    ann_path = utils.get_cache_path(ann_path)
                abs_ann_paths.append(ann_path)
            ann_paths = abs_ann_paths

            # visual data storage path
            vis_path = vis_info.storage

            if not os.path.isabs(vis_path):
                # vis_path = os.path.join(utils.get_cache_path(), vis_path)
                vis_path = utils.get_cache_path(vis_path)

            if not os.path.exists(vis_path):
                warnings.warn("storage path {} does not exist.".format(vis_path))

            # create datasets
            dataset_cls = self.train_dataset_cls if is_train else self.eval_dataset_cls
            datasets[split] = dataset_cls(
                vis_processor=vis_processor,
                text_processor=text_processor,
                ann_paths=ann_paths,
                vis_root=vis_path,
            )

        return datasets


class MultiModalDatasetBuilder(BaseDatasetBuilder):
    """
    MultiModalDatasetBuilder is a utility class designed to construct datasets
    suitable for multi-modal tasks. This class simplifies the creation of 
    datasets that incorporate data of multiple modalities, such as text, 
    images, video, or audio.
    """
    train_dataset_cls, eval_dataset_cls = None, None

    def __init__(self, cfg=None):
        super().__init__(cfg)
        if isinstance(self.data_type, str):
            self.data_type = [self.data_type]

    def _build_processor(self, cfg_name):
        cfg = self.config.get(cfg_name)
        return {
            split: self._build_proc_from_cfg(cfg.get(split)) 
            if cfg is not None 
            else None
            for split in ['train', 'eval']
        }

    def build_processors(self):
        self.text_processors = self._build_processor("text_processor")
        
        self.processors = {
            split: {
                modality: self._build_proc_from_cfg(
                    self.config.get(f"{'vis' if 'image' in modality else modality}_processor").get(split)
                )
                for modality in self.data_type
            }
            for split in ['train', 'eval']
        }

    def _download_multimodal(self, modality):
        storage_path = utils.get_cache_path(self.config.build_info.get(modality).storage)
        if not os.path.exists(storage_path):
            warnings.warn(f"The specified path {storage_path} for {modality} inputs does not exist.")

    def _download_data(self):
        self._download_ann()
        for modality in self.data_type:
            self._download_multimodal(modality)

    def _get_absolute_path(self, path):
        if not os.path.isabs(path):
            return utils.get_cache_path(path)
        return path

    def build(self):
        self.build_processors()
        build_info = self.config.build_info
        datasets = {}
        
        for split, info in build_info.annotations.items():
            if split not in ["train", "val", "test"]:
                continue

            is_train = split == "train"
            dataset_args = self._get_dataset_args(info, is_train)
            
            dataset_cls = self.train_dataset_cls if is_train else self.eval_dataset_cls
            datasets[split] = dataset_cls(**dataset_args)

        return datasets

    def _get_dataset_args(self, info, is_train):
        dataset_args = dict(self.config.build_info.get('kwargs', {}))
        
        for modality in self.data_type:
            proc_name = f"{'vis' if 'image' in modality else modality}_processor"
            dataset_args[proc_name] = self.processors["train" if is_train else "eval"][modality]
            mm_path = self._get_absolute_path(self.config.build_info.get(modality).storage)
            dataset_args[f"{'vis' if 'image' in modality  else modality}_root"] = mm_path
        
        dataset_args['text_processor'] = self.text_processors["train" if is_train else "eval"]
        dataset_args["ann_paths"] = [self._get_absolute_path(path) for path in info.storage]
        dataset_args['modalities'] = self.data_type
        
        # Conform to base
        for key in ['vis_processor', 'vis_root', 'test_processor']:
            dataset_args.setdefault(key, None)
        
        return dataset_args

def load_dataset_config(cfg_path):
    cfg = OmegaConf.load(cfg_path).datasets
    return next(iter(cfg.values()))