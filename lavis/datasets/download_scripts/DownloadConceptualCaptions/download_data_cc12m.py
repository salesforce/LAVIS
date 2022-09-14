"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import time
from PIL import Image
from lavis.common.utils import get_abs_path, get_cache_path
from multiprocessing import Pool
from omegaconf import OmegaConf
from pathlib import Path
from torchvision.transforms import functional as TF
from tqdm import tqdm
import glob
import io
import json
import magic  # pip install python-magic
import numpy as np
import os
import pandas as pd
import requests
import shelve
import zlib

headers = {
    #'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36',
    "User-Agent": "Googlebot-Image/1.0",  # Pretend to be googlebot
    "X-Forwarded-For": "64.18.15.200",
}


def _df_split_apply(tup_arg):
    split_ind, subset, func = tup_arg
    r = subset.apply(func, axis=1)
    return (split_ind, r)


def df_multiprocess(df, processes, chunk_size, func, dataset_name):
    print("Generating parts...")
    with shelve.open(
        "%s_%s_%s_results.tmp" % (dataset_name, func.__name__, chunk_size)
    ) as results:

        pbar = tqdm(total=len(df), position=0)
        # Resume:
        finished_chunks = set([int(k) for k in results.keys()])
        pbar.desc = "Resuming"
        for k in results.keys():
            pbar.update(len(results[str(k)][1]))

        pool_data = (
            (index, df[i : i + chunk_size], func)
            for index, i in enumerate(range(0, len(df), chunk_size))
            if index not in finished_chunks
        )
        print(
            int(len(df) / chunk_size),
            "parts.",
            chunk_size,
            "per part.",
            "Using",
            processes,
            "processes",
        )

        pbar.desc = "Downloading"
        with Pool(processes) as pool:
            for i, result in enumerate(
                pool.imap_unordered(_df_split_apply, pool_data, 2)
            ):
                results[str(result[0])] = result
                pbar.update(len(result[1]))
        pbar.close()

    print("Finished Downloading.")
    return


# Unique name based on url
def _file_name(row):
    name = (
        "%s/%s_%s"
        % (
            # row["folder"],
            storage_dir,
            row.name,
            (zlib.crc32(row["url"].encode("utf-8")) & 0xFFFFFFFF),
        )
        + ".jpg"
    )
    return name


# For checking mimetypes separately without download
def check_mimetype(row):
    if os.path.isfile(str(row["file"])):
        row["mimetype"] = magic.from_file(row["file"], mime=True)
        row["size"] = os.stat(row["file"]).st_size
    return row


# Don't download image, just check with a HEAD request, can't resume.
# Can use this instead of download_image to get HTTP status codes.
def check_download(row):
    fname = _file_name(row)
    try:
        # not all sites will support HEAD
        response = requests.head(
            row["url"], stream=False, timeout=5, allow_redirects=True, headers=headers
        )
        row["status"] = response.status_code
        row["headers"] = dict(response.headers)
    except:
        # log errors later, set error as 408 timeout
        row["status"] = 408
        return row
    if response.ok:
        row["file"] = fname
    return row


def resize_img(req):
    image = Image.open(req).convert("RGB")
    image = TF.resize(
        # image, size=(resize_size, resize_size)
        image,
        size=resize_size,
    )  # , interpolation=Image.LANCZOS)
    return image


def download_image(row):
    fname = _file_name(row)
    # Skip Already downloaded, retry others later
    if os.path.isfile(fname):
        row["status"] = 200
        row["file"] = fname
        row["mimetype"] = magic.from_file(row["file"], mime=True)
        row["size"] = os.stat(row["file"]).st_size
        return row

    try:
        # use smaller timeout to skip errors, but can result in failed downloads
        response = requests.get(
            row["url"], stream=False, timeout=5, allow_redirects=True, headers=headers
        )
        row["status"] = response.status_code
        # row['headers'] = dict(response.headers)
    except Exception as e:
        # log errors later, set error as 408 timeout
        row["status"] = 408
        return row

    if response.ok:
        try:
            # some sites respond with gzip transport encoding
            response.raw.decode_content = True
            img = resize_img(io.BytesIO(response.content))
            img.save(fname)

            row["mimetype"] = magic.from_file(fname, mime=True)
            row["size"] = os.stat(fname).st_size

        except Exception as e:
            #     # This is if it times out during a download or decode
            row["status"] = 408

    row["file"] = fname
    return row


def open_tsv(fname, folder):
    print("Opening %s Data File..." % fname)
    df = pd.read_csv(
        fname, sep="\t", names=["url", "caption"]
    )  # , usecols=range(1, 2))
    df["folder"] = folder
    print("Processing", len(df), " Images:")
    return df


def df_from_shelve(chunk_size, func, dataset_name):
    print("Generating Dataframe from results...")
    with shelve.open(
        "%s_%s_%s_results.tmp" % (dataset_name, func.__name__, chunk_size)
    ) as results:
        keylist = sorted([int(k) for k in results.keys()])
        df = pd.concat([results[str(k)][1] for k in keylist], sort=True)
    return df


resize_size = 384

config_path = get_abs_path("configs/datasets/conceptual_caption/defaults_12m.yaml")

storage_dir = OmegaConf.load(
    config_path
).datasets.conceptual_caption_12m.build_info.images.storage
storage_dir = Path(get_cache_path(storage_dir))

os.makedirs(storage_dir, exist_ok=True)

# number of processes in the pool can be larger than cores
num_processes = 96
# num_processes = 1
# chunk_size is how many images per chunk per process - changing this resets progress when restarting.
images_per_part = 100

data_name = "cc12m"
# os.makedirs(data_name, exist_ok=True)

df = open_tsv("cc12m.tsv", data_name)
df_multiprocess(
    df=df,
    processes=num_processes,
    chunk_size=images_per_part,
    func=download_image,
    dataset_name=data_name,
)
df = df_from_shelve(
    chunk_size=images_per_part, func=download_image, dataset_name=data_name
)
df.to_csv(
    "downloaded_%s_report.tsv.gz" % data_name,
    compression="gzip",
    sep="\t",
    header=False,
    index=False,
)
print("Saved.")
