import os
from pathlib import Path

from omegaconf import OmegaConf

from lavis.common.utils import (
    cleanup_dir,
    download_and_extract_archive,
    get_abs_path,
    get_cache_path,
)


DATA_URL = "https://www.cs.utexas.edu/users/ml/clamp/videoDescription/YouTubeClips.tar"


def download_datasets(root, url):
    download_and_extract_archive(url=url, download_root=root)


def move_files(download_path, storage_path):
    """
    Move files from download_path to storage_path
    """
    print("Moving to {}".format(storage_path))

    os.makedirs(storage_path, exist_ok=True)

    for file_name in os.listdir(download_path):
        os.rename(
            os.path.join(download_path, file_name),
            os.path.join(storage_path, file_name),
        )


if __name__ == "__main__":

    config_path = get_abs_path("configs/datasets/msvd/defaults_cap.yaml")

    storage_dir = OmegaConf.load(
        config_path
    ).datasets.msvd_cap.build_info.videos.storage

    download_dir = Path(get_cache_path(storage_dir)).parent / "download"
    storage_dir = Path(get_cache_path(storage_dir))

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
        print("Downloading {}".format(DATA_URL))
        download_datasets(download_dir, DATA_URL)
    except Exception as e:
        # remove download dir if failed
        cleanup_dir(download_dir)
        print("Failed to download or extracting datasets. Aborting.")

    move_files(download_dir / "YouTubeClips", storage_dir)
    cleanup_dir(download_dir)
