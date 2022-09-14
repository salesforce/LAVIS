"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import json
import logging
import os
import time
from multiprocessing import Pool

import numpy as np
import requests
import tqdm
from lavis.common.utils import cleanup_dir, get_abs_path, get_cache_path
from omegaconf import OmegaConf

header_mzl = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36",
    # "User-Agent": "Googlebot-Image/1.0",  # Pretend to be googlebot
    # "X-Forwarded-For": "64.18.15.200",
}

header_gbot = {
    "User-Agent": "Googlebot-Image/1.0",  # Pretend to be googlebot
}

headers = [header_mzl, header_gbot]

# Setup
logging.basicConfig(filename="download_nocaps.log", filemode="w", level=logging.INFO)
requests.packages.urllib3.disable_warnings(
    requests.packages.urllib3.exceptions.InsecureRequestWarning
)


def download_file(url, filename):
    max_retries = 20
    cur_retries = 0

    header = headers[0]

    while cur_retries < max_retries:
        try:
            r = requests.get(url, headers=header, timeout=10)
            with open(filename, "wb") as f:
                f.write(r.content)

            break
        except Exception as e:
            logging.info(" ".join(repr(e).splitlines()))
            logging.error(url)
            cur_retries += 1

            # random sample a header from headers
            header = headers[np.random.randint(0, len(headers))]

    time.sleep(3 + cur_retries * 2)


def download_image_from_url_val(url):
    basename = os.path.basename(url)
    filename = os.path.join(storage_dir, "val", basename)

    download_file(url, filename)


def download_image_from_url_test(url):
    basename = os.path.basename(url)
    filename = os.path.join(storage_dir, "test", basename)

    download_file(url, filename)


if __name__ == "__main__":
    os.makedirs("tmp", exist_ok=True)

    # storage dir
    config_path = get_abs_path("configs/datasets/nocaps/defaults.yaml")

    storage_dir = OmegaConf.load(config_path).datasets.nocaps.build_info.images.storage
    storage_dir = get_cache_path(storage_dir)
    # make sure the storage dir exists
    os.makedirs(storage_dir, exist_ok=True)
    print("Storage dir:", storage_dir)

    # make sure the storage dir for val and test exists
    os.makedirs(os.path.join(storage_dir, "val"), exist_ok=True)
    os.makedirs(os.path.join(storage_dir, "test"), exist_ok=True)

    # download annotations
    val_url = "https://nocaps.s3.amazonaws.com/nocaps_val_4500_captions.json"
    tst_url = "https://s3.amazonaws.com/nocaps/nocaps_test_image_info.json"

    print("Downloading validation annotations from %s" % val_url)
    download_file(val_url, "tmp/nocaps_val_ann.json")
    print("Downloading testing annotations from %s" % tst_url)
    download_file(tst_url, "tmp/nocaps_tst_ann.json")

    # open annotations
    val_ann = json.load(open("tmp/nocaps_val_ann.json"))
    tst_ann = json.load(open("tmp/nocaps_tst_ann.json"))

    # collect image urls
    val_info = val_ann["images"]
    tst_info = tst_ann["images"]

    val_urls = [info["coco_url"] for info in val_info]
    tst_urls = [info["coco_url"] for info in tst_info]

    # setup multiprocessing
    # large n_procs possibly causes server to reject requests
    n_procs = 16

    with Pool(n_procs) as pool:
        print("Downloading validation images...")
        list(
            tqdm.tqdm(
                pool.imap(download_image_from_url_val, val_urls), total=len(val_urls)
            )
        )

    with Pool(n_procs) as pool:
        print("Downloading test images...")
        list(
            tqdm.tqdm(
                pool.imap(download_image_from_url_test, tst_urls), total=len(tst_urls)
            )
        )

    # clean tmp
    cleanup_dir("tmp")
