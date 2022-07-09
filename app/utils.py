import streamlit as st
import torch
import numpy as np

from PIL import Image

from matplotlib import pyplot as plt
from scipy.ndimage import filters
from skimage import transform as skimage_transform

from lavis.models.blip_models import init_tokenizer
from lavis.models import BlipITM, load_model


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
    tokenizer = init_tokenizer()
    return tokenizer


def compute_gradcam(model, visual_input, text_input, tokenized_text, block_num=6):
    model.text_encoder.base_model.base_model.encoder.layer[
        block_num
    ].crossattention.self.save_attention = True

    output = model(visual_input, text_input, match_head="itm")
    loss = output[:, 1].sum()

    model.zero_grad()
    loss.backward()
    with torch.no_grad():
        mask = tokenized_text.attention_mask.view(
            tokenized_text.attention_mask.size(0), 1, -1, 1, 1
        )  # (bsz,1,token_len, 1,1)
        token_length = mask.sum() - 2
        token_length = token_length.cpu()
        # grads and cams [bsz, num_head, seq_len, image_patch]
        grads = model.text_encoder.base_model.base_model.encoder.layer[
            block_num
        ].crossattention.self.get_attn_gradients()
        cams = model.text_encoder.base_model.base_model.encoder.layer[
            block_num
        ].crossattention.self.get_attention_map()

        # assume using vit large with 576 num image patch
        cams = cams[:, :, :, 1:].reshape(visual_input.size(0), 12, -1, 24, 24) * mask
        grads = (
            grads[:, :, :, 1:].clamp(0).reshape(visual_input.size(0), 12, -1, 24, 24)
            * mask
        )

        gradcam = cams * grads
        gradcam = gradcam[0].mean(0).cpu().detach()
        # [enc token gradcam, average gradcam across token, gradcam for individual token]
        gradcam = torch.cat(
            (
                gradcam[0:1, :],
                gradcam[1 : token_length + 1, :].sum(dim=0, keepdim=True)
                / token_length,
                gradcam[1:, :],
            )
        )

    return gradcam, output


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
    if model_type == "large":
        pretrained_path = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_retrieval_coco.pth"
    else:
        pretrained_path = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_retrieval_coco.pth"
    model = BlipITM(pretrained=pretrained_path, vit=model_type)
    model.eval()
    model = model.to(device)
    return model
