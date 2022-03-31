from datasets.builders.coco_builder import COCOBuilder

from common.registry import registry


@registry.register_builder("coco_vqa")
class COCOVQABuilder(COCOBuilder):
    train_dataset_cls = None
    eval_dataset_cls = None

    def __init__(self, cfg):
        super().__init__(cfg)
    
    @classmethod
    def default_config_path(cls):
        return "configs/datasets/vqa/defaults.yaml"
    
    # def _download_vis(self):
    #     vis_sources = self.config.build_info.get(self.data_type)

    #     splits = vis_sources.keys()

    #     cache_root = registry.get_path('cache_root')

    #     # check whether has been fully built
    #     # Create temp directory for caching.
    #     dl_cache_dir = os.path.join(cache_root, 'temp/coco')
    #     os.makedirs(dl_cache_dir, exist_ok=True)

    #     build_info_dir = os.path.join(cache_root, 'build_info')
    #     os.makedirs(build_info_dir, exist_ok=True)

    #     # Download *.zip files
    #     for split in splits:
    #         info = vis_sources[split]

    #         urls, storage_paths = info.url, info.storage

    #         if isinstance(urls, str): urls = [urls]
    #         if isinstance(storage_paths, str): storage_paths = [storage_paths]

    #         for url, storage_path in zip(urls, storage_paths):
    #             # if storage_path is relative, make it full by prefixing with cache_root.
    #             if not os.path.isabs(storage_path):
    #                 storage_path = os.path.join(cache_root, storage_path)

    #             build_info_path = os.path.join(build_info_dir, "coco_" + os.path.splitext(os.path.basename(url))[0] + '.build')

    #             if os.path.exists(build_info_path):
    #                 logging.info("Path {} exists, skip downloading.".format(build_info_path))
    #                 continue

    #             # download_url(url=remote_paths[split].url, root=dl_cache_dir, md5=remote_paths[split].md5)
    #             download_url(url=url, root=dl_cache_dir)

    #             dirname = os.path.dirname(storage_path)
    #             assert os.path.normpath(dirname) == os.path.normpath(storage_path), "Local path to store images has to be a directory, found {}.".format(storage_path)

    #             if not os.path.exists(dirname): os.makedirs(dirname)
                
    #             # extracting
    #             archive_path = os.path.join(dl_cache_dir, os.path.basename(url))
    #             extract_archive(from_path=archive_path, to_path=storage_path, overwrite=False)

    #             # save build info
    #             self.save_build_info(build_info_path, url)
    
    def build(self):
        import pdb; pdb.set_trace()
        pass