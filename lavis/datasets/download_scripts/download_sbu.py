"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import pathlib
from omegaconf import OmegaConf

from lavis.common.utils import (
    cleanup_dir,
    download_and_extract_archive,
    get_abs_path,
    get_cache_path,
)


DATA_URL = {"train": "http://www.cs.rice.edu/~vo9/sbucaptions/sbu_images.tar"}


def download_datasets(root, url):
    download_and_extract_archive(
        url=url, download_root=root, extract_root=storage_dir.parent
    )


if __name__ == "__main__":

    config_path = get_abs_path("configs/datasets/sbu_caption/defaults.yaml")

    storage_dir = OmegaConf.load(
        config_path
    ).datasets.sbu_caption.build_info.images.storage

    download_dir = pathlib.Path(get_cache_path(storage_dir)).parent / "download"
    storage_dir = pathlib.Path(get_cache_path(storage_dir))

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

    cleanup_dir(download_dir)
