"""
 # Copyright (c) 2022, salesforce.com, inc.
 # All rights reserved.
 # SPDX-License-Identifier: BSD-3-Clause
 # For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import numpy as np
import streamlit as st
import torch
from lavis.models.blip_models.blip_image_text_matching import compute_gradcam
from lavis.processors import load_processor
from PIL import Image

from app import device, load_demo_image
from app.utils import getAttMap, init_bert_tokenizer, load_blip_itm_model


def app():
    model_type = st.sidebar.selectbox("Model:", ["BLIP_base", "BLIP_large"])

    if model_type.startswith("BLIP"):
        blip_type = model_type.split("_")[1]
        model = load_blip_itm_model(device, model_type=blip_type)

    vis_processor = load_processor("blip_image_eval").build(image_size=384)

    st.markdown(
        "<h1 style='text-align: center;'>Image Text Matching</h1>",
        unsafe_allow_html=True,
    )

    values = list(range(1, 12))
    default_layer_num = values.index(7)
    layer_num = (
        st.sidebar.selectbox("Layer number", values, index=default_layer_num) - 1
    )

    instructions = """Try the provided image or upload your own:"""
    file = st.file_uploader(instructions)

    col1, col2 = st.columns(2)
    col1.header("Image")
    col2.header("GradCam")
    if file:
        raw_img = Image.open(file).convert("RGB")
    else:
        raw_img = load_demo_image()

    w, h = raw_img.size
    scaling_factor = 720 / w
    resized_image = raw_img.resize((int(w * scaling_factor), int(h * scaling_factor)))
    col1.image(resized_image, use_column_width=True)

    col3, col4 = st.columns(2)
    col3.header("Text")
    user_question = col3.text_input(
        "Input your sentence!", "a woman sitting on the beach with a dog"
    )
    submit_button = col3.button("Submit")

    col4.header("Matching score")

    if submit_button:
        tokenizer = init_bert_tokenizer()

        img = vis_processor(raw_img).unsqueeze(0).to(device)
        text_processor = load_processor("blip_caption").build()

        qry = text_processor(user_question)

        norm_img = np.float32(resized_image) / 255

        qry_tok = tokenizer(qry, return_tensors="pt").to(device)
        gradcam, output = compute_gradcam(model, img, qry, qry_tok, block_num=layer_num)

        avg_gradcam = getAttMap(norm_img, gradcam[0][1], blur=True)

        col2.image(avg_gradcam, use_column_width=True, clamp=True)
        # output = model(img, question)
        itm_score = torch.nn.functional.softmax(output, dim=1)
        new_title = (
            '<p style="text-align: left; font-size: 25px;">\n{:.3f}%</p>'.format(
                itm_score[0][1].item() * 100
            )
        )
        col4.markdown(new_title, unsafe_allow_html=True)
