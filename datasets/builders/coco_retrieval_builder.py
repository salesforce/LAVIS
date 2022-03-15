import os
import logging

from common.registry import registry
from datasets.builders.retrieval_builder import RetrievalBuilder

from torchvision.datasets.utils import download_url

from utils.file_utils import extract_archive


@registry.register_builder("coco_retrieval")
class COCORetrievalBuilder(RetrievalBuilder):
    def __init__(self, cfg):
        super().__init__(cfg)
    
    @classmethod
    def default_config_path(cls):
        return "configs/datasets/coco/defaults.yaml"
    
    def _download_vis(self):
        local_paths = self.config.storage.get(self.data_type)
        remote_paths = self.config.build_info.get(self.data_type)

        local_splits = local_paths.keys()
        remote_splits = remote_paths.keys()

        assert local_splits == remote_splits, "Inconsistent remote and local splits, found {} and {}.".format(remote_splits, local_splits)

        cache_root = registry.get_path('cache_root')

        # check whether has been fully built
        # Create temp directory for caching.
        dl_cache_dir = os.path.join(cache_root, 'temp')
        os.makedirs(dl_cache_dir, exist_ok=True)

        # Download *.zip files
        for split in local_splits:
            # extract the downloaded archive files.
            storage_path = local_paths[split]

            # if storage_path is relative, make it full by prefixing with cache_root.
            if not os.path.isabs(storage_path):
                storage_path = os.path.join(cache_root, storage_path)

            if os.path.exists(storage_path):
                logging.info("Path {} exists, skip downloading.".format(storage_path))
                continue

            download_url(url=remote_paths[split].url, root=dl_cache_dir, md5=remote_paths[split].md5)

            dirname = os.path.dirname(storage_path)
            assert os.path.normpath(dirname) == os.path.normpath(storage_path), "Local path to store COCO images has to be a directory, found {}.".format(storage_path)

            if not os.path.exists(dirname): os.makedirs(dirname)
            
            # extracting
            archive_path = os.path.join(dl_cache_dir, os.path.basename(remote_paths[split].url))
            extract_archive(from_path=archive_path, to_path=storage_path, overwrite=False)