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
    download_and_extract_archive(url=url, download_root=root, extract_root=storage_dir)


if __name__ == "__main__":

    config_path = get_abs_path("configs/datasets/sbu_caption/defaults.yaml")

    storage_dir = OmegaConf.load(
        config_path
    ).datasets.sbu_caption.build_info.images.storage

    download_dir = pathlib.Path(get_cache_path(storage_dir)).parent / "download"
    storage_dir = pathlib.Path(get_cache_path(storage_dir))

    if storage_dir.exists():
        # ask users to confirm
        ans = input(
            "{} exists. Do you want to delete it and re-download? [y/N] ".format(
                storage_dir
            )
        )

        if ans in ["y", "Y", "yes", "Yes"]:
            cleanup_dir(storage_dir)
            cleanup_dir(download_dir)
            os.makedirs(download_dir)
        else:
            print("Aborting")
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
