import streamlit as st
import torch
import numpy as np
from app import pending_job_path

import os, glob

from PIL import Image

from matplotlib import pyplot as plt
from scipy.ndimage import filters
from skimage import transform as skimage_transform

from lavis.models import BlipBase, load_model

def get_pending_jobs(job_type):
    list_of_prompts = filter(os.path.isfile,
                        glob.glob('{}/{}/*.txt'.format(pending_job_path, job_type) ))
    # Sort list of files based on last modification time in ascending order
    list_of_prompts = sorted(list_of_prompts,
                       key = os.path.getmtime)
    return list(list_of_prompts)

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
    # if model_type == "large":
    #     pretrained_path = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_retrieval_coco.pth"
    # else:
    #     pretrained_path = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_retrieval_coco.pth"
    # model = BlipITM(pretrained=pretrained_path, vit=model_type)
    # model.eval()
    # model = model.to(device)
    # return model
    model = load_model(
        "blip_image_text_matching", model_type, is_eval=True, device=device
    )
    return model

def create_uniq_user_job_name(time_stamp, user_info):
    return str(time_stamp).replace('.','_') + '_' + '_'.join(user_info.split(' ')[:20])
