import os
from pathlib import Path

from omegaconf import OmegaConf

from lavis.common.utils import (
    cleanup_dir,
    get_abs_path,
    get_cache_path,
)

import opendatasets as od


DATA_URL = "https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset"

print(
    """
    To download the dataset, you need to have a Kaggle account and the associated key.
    See https://www.kaggle.com/docs/api to create account and a new API token.
    """
)


def move_directory(src_dir, dst_dir):
    """
    Move files from download_path to storage_path
    """
    print("Moving to {}".format(dst_dir))

    os.makedirs(dst_dir, exist_ok=True)

    for file_name in os.listdir(src_dir):
        os.rename(
            os.path.join(src_dir, file_name),
            os.path.join(dst_dir, file_name),
        )


if __name__ == "__main__":

    config_path = get_abs_path("configs/datasets/flickr30k/defaults.yaml")

    storage_dir = OmegaConf.load(
        config_path
    ).datasets.flickr30k.build_info.images.storage

    storage_dir = Path(get_cache_path(storage_dir))
    download_dir = storage_dir.parent / "download"

    if storage_dir.exists():
        # ask users to confirm
        ans = input(
            "{} exists. Do you want to delete it and re-download? [y/N] ".format(
                storage_dir
            )
        )

        if ans in ["y", "Y", "yes", "Yes"]:
            cleanup_dir(storage_dir)
            if download_dir.exists():
                cleanup_dir(download_dir)
        else:
            print("Aborting")
            exit(1)

    os.makedirs(download_dir)

    try:
        print("Downloading {} to {}".format(DATA_URL, download_dir))
        od.download(DATA_URL, download_dir)
    except Exception as e:
        print(e)
        # remove download dir if failed
        cleanup_dir(download_dir)
        exit(1)

    move_directory(
        download_dir / "flickr-image-dataset" / "flickr30k_images" / "flickr30k_images",
        storage_dir / "flickr30k-images",
    )

    cleanup_dir(download_dir)
