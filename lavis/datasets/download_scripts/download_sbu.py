"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import io
import os
import pathlib
import urllib
import tqdm

from concurrent.futures import ThreadPoolExecutor

from lavis.common.utils import get_abs_path, get_cache_path
from lavis.datasets.builders import load_dataset
from omegaconf import OmegaConf
from PIL import Image

# DATA_URL = {"train": "http://www.cs.rice.edu/~vo9/sbucaptions/sbu_images.tar"}

USER_AGENT = (
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:15.0) Gecko/20100101 Firefox/15.0.1"
)


def fetch_single_image(image_url, timeout=None, retries=0):
    for _ in range(retries + 1):
        try:
            request = urllib.request.Request(
                image_url,
                data=None,
                headers={"user-agent": USER_AGENT},
            )
            with urllib.request.urlopen(request, timeout=timeout) as req:
                image = Image.open(io.BytesIO(req.read()))
            break
        except Exception:
            image = None
    return image


def download_and_save_image(ann, save_dir, timeout=None, retries=0):
    image = fetch_single_image(ann["url"], timeout=timeout, retries=retries)

    if image is not None:
        image_path = os.path.join(save_dir, ann["image"])
        print(image_path)
        image.save(image_path)


if __name__ == "__main__":

    config_path = get_abs_path("configs/datasets/sbu_caption/defaults.yaml")

    storage_dir = OmegaConf.load(
        config_path
    ).datasets.sbu_caption.build_info.images.storage

    storage_dir = pathlib.Path(get_cache_path(storage_dir))

    if storage_dir.exists():
        print(f"Dataset already exists at {storage_dir}. Aborting.")
        exit(0)

    storage_dir.mkdir(parents=True, exist_ok=True)

    num_threads = 20
    dset = load_dataset("sbu_caption")["train"].annotation

    print("Downloading dataset...")
    # multiprocessing
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        for ann in tqdm.tqdm(dset):
            executor.submit(
                download_and_save_image,
                ann,
                storage_dir,
                timeout=30,
                retries=10,
            )
