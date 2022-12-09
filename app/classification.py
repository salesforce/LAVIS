import plotly.graph_objects as go
import requests
import streamlit as st
import torch
from lavis.models import BlipFeatureExtractor, load_model
from lavis.models.blip_models.blip_image_text_matching import BlipITM
from lavis.processors import load_processor
from lavis.processors.blip_processors import BlipCaptionProcessor
from PIL import Image
import numpy as np

from app import device, load_demo_image
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


@st.cache(
    hash_funcs={
        torch.nn.parameter.Parameter: lambda parameter: parameter.data.detach()
        .cpu()
        .numpy()
    },
    allow_output_mutation=True,
)
def load_blip_itm_model(device):
    pretrained_path = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_retrieval_coco.pth"
    model = BlipITM(pretrained=pretrained_path, vit="base")
    model.eval()
    model = model.to(device)
    return model


def app():
    model_type = st.sidebar.selectbox(
        "Model:",
        ["BLIP_Base", "CLIP_ViT-B-32", "CLIP_ViT-B-16", "CLIP_ViT-L-14"],
    )
    score_type = st.sidebar.selectbox("Score type:", ["Cosine", "Multimodal"])

    # ===== layout =====
    st.markdown(
        " <h1 style='text-align: center;'>Zero-shot Classification</h1>",
        unsafe_allow_html=True,
    )

    instructions = """Try the provided image or upload your own:"""
    file = st.file_uploader(instructions)

    col1,col2 = st.columns(2)
    col1.header("Image")
    #col2.header("Categories")
    row2_col1,row2_col2 = st.columns(2)
    if file:
        raw_img = Image.open(file).convert("RGB")
        st.session_state.new_image = 'yes'
        st.session_state.category = 'yes'
    else:
        raw_img = load_demo_image()

    w, h = raw_img.size
    scaling_factor = 700 / w
    resized_image = raw_img.resize((int(w * scaling_factor), int(h * scaling_factor)))

    row2_col1.image(resized_image, use_column_width=True)

    cls_names = []
    button = st.button("Submit")
    if 'cls_names' not in st.session_state or st.session_state.cls_names == '' or not button:
        col2.header("Categories")
        with row2_col2:
            cls_0 = st.text_input("category 1", value="merlion")
            cls_1 = st.text_input("category 2", value="elephant")
            cls_2 = st.text_input("category 3", value="giraffe")
            cls_3 = st.text_input("category 4", value="fountain")
            cls_4 = st.text_input("category 5", value="marina bay")
        cls_names = [cls_0, cls_1, cls_2, cls_3, cls_4 ]
        cls_names = [cls_nm for cls_nm in cls_names if len(cls_nm) > 0]
        st.session_state.cls_names = ','.join(cls_names)

        if len(cls_names) != len(set(cls_names)):
            st.error("Please provide unique class names")
            return
    # ===== event =====
    if button:
        if model_type.startswith("BLIP"):
            text_processor = BlipCaptionProcessor(prompt="A picture of ")
            cls_names = st.session_state['cls_names'].split(',')
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

                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    text_features /= text_features.norm(dim=-1, keepdim=True)

                    sims = (100.0 * image_features @ text_features.T)[0].softmax(dim=-1)
                    inv_sims = sims.tolist()[::-1]
            else:
                st.warning("CLIP does not support multimodal scoring.")
                return

        fig = go.Figure(
            go.Bar(
                x=inv_sims,
                y=[c+' ' for c in cls_names[::-1]],
                text=["{:.2f}%".format(s) for s in inv_sims],
                orientation="h",
            )
        )
        fig.update_traces(
            textfont_size=16,
            textangle=0,
            textposition="outside",
            cliponaxis=False,
            marker_color="#0176D3"
        )
        fig.add_vline(x=0, line_width=1, line_color="#C9C9C9")
        fig.add_hline(y=-0.6, line_width=1, line_color="#C9C9C9")
        fig.update_layout(font=dict(family="Salesforce Sans", size=25, color="#032D60"))
        fig.update_layout(
                  xaxis = dict(
                    tickmode='linear',
                    tickfont = dict(size=16),
                    tick0=0,
                    dtick=20,
                    ticksuffix="%"
                    ),
                   yaxis = dict(tickfont = dict(size=16)),
                   plot_bgcolor= "rgba(0, 0, 0, 0)",
                   title="<b>Zero-shot image classification</b>",
                   title_font_family="Salesforce Sans",
                   title_font_size=28,
                   title_font_color="#032D60",
                   paper_bgcolor= "rgba(0, 0, 0, 0)",)
        fig.update_xaxes(fixedrange=True, dtick=20)
        fig.update_xaxes(range=[0, 100], ticks="outside", tickson="boundaries", ticklen=6)
        col2.header("Prediction")
        with row2_col2:
            st.plotly_chart(fig, use_container_width=True)
