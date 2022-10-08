"""
 # Copyright (c) 2022, salesforce.com, inc.
 # All rights reserved.
 # SPDX-License-Identifier: BSD-3-Clause
 # For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import plotly.graph_objects as go
import requests
import streamlit as st
import torch
from lavis.models import load_model
from lavis.processors import load_processor
from lavis.processors.blip_processors import BlipCaptionProcessor
from PIL import Image

from app import device, load_demo_image
from app.utils import load_blip_itm_model
from lavis.processors.clip_processors import ClipImageEvalProcessor


@st.cache()
def load_demo_image(img_url=None):
    if not img_url:
        img_url = "https://img.atlasobscura.com/yDJ86L8Ou6aIjBsxnlAy5f164w1rjTgcHZcx2yUs4mo/rt:fit/w:1200/q:81/sm:1/scp:1/ar:1/aHR0cHM6Ly9hdGxh/cy1kZXYuczMuYW1h/em9uYXdzLmNvbS91/cGxvYWRzL3BsYWNl/X2ltYWdlcy85MDll/MDRjOS00NTJjLTQx/NzQtYTY4MS02NmQw/MzI2YWIzNjk1ZGVk/MGZhMTJiMTM5MmZi/NGFfUmVhcl92aWV3/X29mX3RoZV9NZXJs/aW9uX3N0YXR1ZV9h/dF9NZXJsaW9uX1Bh/cmssX1NpbmdhcG9y/ZSxfd2l0aF9NYXJp/bmFfQmF5X1NhbmRz/X2luX3RoZV9kaXN0/YW5jZV8tXzIwMTQw/MzA3LmpwZw.jpg"
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
    return raw_image


@st.cache(
    hash_funcs={
        torch.nn.parameter.Parameter: lambda parameter: parameter.data.detach()
        .cpu()
        .numpy()
    },
    allow_output_mutation=True,
)
def load_model_cache(model_type, device):
    if model_type == "blip":
        model = load_model(
            "blip_feature_extractor", model_type="base", is_eval=True, device=device
        )
    elif model_type == "albef":
        model = load_model(
            "albef_feature_extractor", model_type="base", is_eval=True, device=device
        )
    elif model_type == "CLIP_ViT-B-32":
        model = load_model(
            "clip_feature_extractor", "ViT-B-32", is_eval=True, device=device
        )
    elif model_type == "CLIP_ViT-B-16":
        model = load_model(
            "clip_feature_extractor", "ViT-B-16", is_eval=True, device=device
        )
    elif model_type == "CLIP_ViT-L-14":
        model = load_model(
            "clip_feature_extractor", "ViT-L-14", is_eval=True, device=device
        )

    return model


def app():
    model_type = st.sidebar.selectbox(
        "Model:",
        ["ALBEF", "BLIP_Base", "CLIP_ViT-B-32", "CLIP_ViT-B-16", "CLIP_ViT-L-14"],
    )
    score_type = st.sidebar.selectbox("Score type:", ["Cosine", "Multimodal"])

    # ===== layout =====
    st.markdown(
        "<h1 style='text-align: center;'>Zero-shot Classification</h1>",
        unsafe_allow_html=True,
    )

    instructions = """Try the provided image or upload your own:"""
    file = st.file_uploader(instructions)

    st.header("Image")
    if file:
        raw_img = Image.open(file).convert("RGB")
    else:
        raw_img = load_demo_image()

    st.image(raw_img)  # , use_column_width=True)

    col1, col2 = st.columns(2)

    col1.header("Categories")

    cls_0 = col1.text_input("category 1", value="merlion")
    cls_1 = col1.text_input("category 2", value="sky")
    cls_2 = col1.text_input("category 3", value="giraffe")
    cls_3 = col1.text_input("category 4", value="fountain")
    cls_4 = col1.text_input("category 5", value="marina bay")

    cls_names = [cls_0, cls_1, cls_2, cls_3, cls_4]
    cls_names = [cls_nm for cls_nm in cls_names if len(cls_nm) > 0]

    if len(cls_names) != len(set(cls_names)):
        st.error("Please provide unique class names")
        return

    button = st.button("Submit")

    col2.header("Prediction")

    # ===== event =====

    if button:
        if model_type.startswith("BLIP"):
            text_processor = BlipCaptionProcessor(prompt="A picture of ")
            cls_prompt = [text_processor(cls_nm) for cls_nm in cls_names]

            if score_type == "Cosine":
                vis_processor = load_processor("blip_image_eval").build(image_size=224)
                img = vis_processor(raw_img).unsqueeze(0).to(device)

                feature_extractor = load_model_cache(model_type="blip", device=device)

                sample = {"image": img, "text_input": cls_prompt}

                with torch.no_grad():
                    image_features = feature_extractor.extract_features(
                        sample, mode="image"
                    ).image_embeds_proj[:, 0]
                    text_features = feature_extractor.extract_features(
                        sample, mode="text"
                    ).text_embeds_proj[:, 0]
                    sims = (image_features @ text_features.t())[
                        0
                    ] / feature_extractor.temp

            else:
                vis_processor = load_processor("blip_image_eval").build(image_size=384)
                img = vis_processor(raw_img).unsqueeze(0).to(device)

                model = load_blip_itm_model(device)

                output = model(img, cls_prompt, match_head="itm")
                sims = output[:, 1]

            sims = torch.nn.Softmax(dim=0)(sims)
            inv_sims = [sim * 100 for sim in sims.tolist()[::-1]]

        elif model_type.startswith("ALBEF"):
            vis_processor = load_processor("blip_image_eval").build(image_size=224)
            img = vis_processor(raw_img).unsqueeze(0).to(device)

            text_processor = BlipCaptionProcessor(prompt="A picture of ")
            cls_prompt = [text_processor(cls_nm) for cls_nm in cls_names]

            feature_extractor = load_model_cache(model_type="albef", device=device)

            sample = {"image": img, "text_input": cls_prompt}

            with torch.no_grad():
                image_features = feature_extractor.extract_features(
                    sample, mode="image"
                ).image_embeds_proj[:, 0]
                text_features = feature_extractor.extract_features(
                    sample, mode="text"
                ).text_embeds_proj[:, 0]

                st.write(image_features.shape)
                st.write(text_features.shape)

                sims = (image_features @ text_features.t())[0] / feature_extractor.temp

            sims = torch.nn.Softmax(dim=0)(sims)
            inv_sims = [sim * 100 for sim in sims.tolist()[::-1]]

        elif model_type.startswith("CLIP"):
            if model_type == "CLIP_ViT-B-32":
                model = load_model_cache(model_type="CLIP_ViT-B-32", device=device)
            elif model_type == "CLIP_ViT-B-16":
                model = load_model_cache(model_type="CLIP_ViT-B-16", device=device)
            elif model_type == "CLIP_ViT-L-14":
                model = load_model_cache(model_type="CLIP_ViT-L-14", device=device)
            else:
                raise ValueError(f"Unknown model type {model_type}")

            if score_type == "Cosine":
                # image_preprocess = ClipImageEvalProcessor(image_size=336)
                image_preprocess = ClipImageEvalProcessor(image_size=224)
                img = image_preprocess(raw_img).unsqueeze(0).to(device)

                sample = {"image": img, "text_input": cls_names}

                with torch.no_grad():
                    clip_features = model.extract_features(sample)

                    image_features = clip_features.image_embeds_proj
                    text_features = clip_features.text_embeds_proj

                    sims = (100.0 * image_features @ text_features.T)[0].softmax(dim=-1)
                    inv_sims = sims.tolist()[::-1]
            else:
                st.warning("CLIP does not support multimodal scoring.")
                return

        fig = go.Figure(
            go.Bar(
                x=inv_sims,
                y=cls_names[::-1],
                text=["{:.2f}".format(s) for s in inv_sims],
                orientation="h",
            )
        )
        fig.update_traces(
            textfont_size=12,
            textangle=0,
            textposition="outside",
            cliponaxis=False,
        )
        col2.plotly_chart(fig, use_container_width=True)
