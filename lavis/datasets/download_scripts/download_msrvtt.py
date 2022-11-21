"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
from pathlib import Path

from omegaconf import OmegaConf

from lavis.common.utils import (
    cleanup_dir,
    download_and_extract_archive,
    get_abs_path,
    get_cache_path,
)


# TODO
# 1. Go to https://www.mediafire.com/file/czh8sezbo9s4692/test_videos.zip/file
#      and https://www.mediafire.com/file/x3rrbe4hwp04e6w/train_val_videos.zip/file
# 2. Right-click the Download button and copy the link address
#      e.g.
#    DATA_URL = {
#        "train": "https://download1602.mediafire.com/xxxxxxxxxxxx/x3rrbe4hwp04e6w/train_val_videos.zip",
#        "test": "https://download2390.mediafire.com/xxxxxxxxxxxx/czh8sezbo9s4692/test_videos.zip",
#    }
# 3. Paste the link address to DATA_URL

DATA_URL = {
    "train": "https://download2295.mediafire.com/4bb7p74xrbgg/x3rrbe4hwp04e6w/train_val_videos.zip",
    "test": "https://download2390.mediafire.com/79hfq3592lqg/czh8sezbo9s4692/test_videos.zip",
}


def download_datasets(root, url):
    """
    Download the Imagenet-R dataset archives and expand them
    in the folder provided as parameter
    """
    download_and_extract_archive(url=url, download_root=root)


def merge_datasets(download_path, storage_path):
    """
    Merge datasets in download_path to storage_path
    """

    # Merge train and test datasets
    train_path = os.path.join(download_path, "TrainValVideo")
    test_path = os.path.join(download_path, "TestVideo")
    train_test_path = storage_path

    print("Merging to {}".format(train_test_path))

    os.makedirs(train_test_path, exist_ok=True)

    for file_name in os.listdir(train_path):
        os.rename(
            os.path.join(train_path, file_name),
            os.path.join(train_test_path, file_name),
        )

    for file_name in os.listdir(test_path):
        os.rename(
            os.path.join(test_path, file_name),
            os.path.join(train_test_path, file_name),
        )


if __name__ == "__main__":

    config_path = get_abs_path("configs/datasets/msrvtt/defaults_cap.yaml")

    storage_dir = OmegaConf.load(
        config_path
    ).datasets.msrvtt_cap.build_info.videos.storage

    download_dir = Path(get_cache_path(storage_dir)).parent / "download"
    storage_dir = Path(get_cache_path(storage_dir))

    if storage_dir.exists():
        print(f"Dataset already exists at {storage_dir}. Aborting.")
        exit(0)

    try:
        for k, v in DATA_URL.items():
            print("Downloading {} to {}".format(v, k))
            download_datasets(download_dir, v)
    except Exception as e:
        # remove download dir if failed
        cleanup_dir(download_dir)
        print("Failed to download or extracting datasets. Aborting.")

    try:
        merge_datasets(download_dir, storage_dir)
    except Exception as e:
        # remove storage dir if failed
        cleanup_dir(download_dir)
        cleanup_dir(storage_dir)
        print("Failed to merging datasets. Aborting.")

    cleanup_dir(download_dir)
