# import os
# import logging

# from torchvision.datasets.utils import download_url

# from datasets.builders.base_dataset_builder import BaseDatasetBuilder

# from utils.file_utils import extract_archive
# from common.registry import registry


# class COCOBuilder(BaseDatasetBuilder, DatasetBuilderDownloadMixin):
#     def __init__(self, cfg):
#         super().__init__(cfg)