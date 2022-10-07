"""
 # Copyright (c) 2022, salesforce.com, inc.
 # All rights reserved.
 # SPDX-License-Identifier: BSD-3-Clause
 # For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import numpy as np
import streamlit as st
import torch
from lavis.models import BlipBase, load_model
from matplotlib import pyplot as plt
from PIL import Image
from scipy.ndimage import filters
from skimage import transform as skimage_transform


def resize_img(raw_img):
    w, h = raw_img.size
    scaling_factor = 240 / w
    resized_image = raw_img.resize((int(w * scaling_factor), int(h * scaling_factor)))
    return resized_image


def read_img(filepath):
    raw_image = Image.open(filepath).convert("RGB")

    return raw_image


@st.cache(
    hash_funcs={
        torch.nn.parameter.Parameter: lambda parameter: parameter.data.detach()
        .cpu()
        .numpy()
    },
    allow_output_mutation=True,
)
def load_model_cache(name, model_type, is_eval, device):
    return load_model(name, model_type, is_eval, device)


@st.cache(allow_output_mutation=True)
def init_bert_tokenizer():
    tokenizer = BlipBase.init_tokenizer()
    return tokenizer


def getAttMap(img, attMap, blur=True, overlap=True):
    attMap -= attMap.min()
    if attMap.max() > 0:
        attMap /= attMap.max()
    attMap = skimage_transform.resize(attMap, (img.shape[:2]), order=3, mode="constant")
    if blur:
        attMap = filters.gaussian_filter(attMap, 0.02 * max(img.shape[:2]))
        attMap -= attMap.min()
        attMap /= attMap.max()
    cmap = plt.get_cmap("jet")
    attMapV = cmap(attMap)
    attMapV = np.delete(attMapV, 3, 2)
    if overlap:
        attMap = (
            1 * (1 - attMap**0.7).reshape(attMap.shape + (1,)) * img
            + (attMap**0.7).reshape(attMap.shape + (1,)) * attMapV
        )
    return attMap


@st.cache(
    hash_funcs={
        torch.nn.parameter.Parameter: lambda parameter: parameter.data.detach()
        .cpu()
        .numpy()
    },
    allow_output_mutation=True,
)
def load_blip_itm_model(device, model_type="base"):
    model = load_model(
        "blip_image_text_matching", model_type, is_eval=True, device=device
    )
    return model
