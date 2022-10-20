"""
 # Copyright (c) 2022, salesforce.com, inc.
 # All rights reserved.
 # SPDX-License-Identifier: BSD-3-Clause
 # For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import math

import numpy as np
import streamlit as st
from lavis.models.blip_models.blip_image_text_matching import compute_gradcam
from lavis.processors import load_processor
from PIL import Image

from app import device, load_demo_image
from app.utils import getAttMap, init_bert_tokenizer, load_blip_itm_model


def app():
    model_type = st.sidebar.selectbox("Model:", ["BLIP_base", "BLIP_large"])

    values = list(range(1, 12))
    default_layer_num = values.index(7)
    layer_num = (
        st.sidebar.selectbox("Layer number", values, index=default_layer_num) - 1
    )

    st.markdown(
        "<h1 style='text-align: center;'>Text Localization</h1>", unsafe_allow_html=True
    )

    vis_processor = load_processor("blip_image_eval").build(image_size=384)
    text_processor = load_processor("blip_caption")

    tokenizer = init_bert_tokenizer()

    instructions = "Try the provided image and text or use your own ones."
    file = st.file_uploader(instructions)

    query = st.text_input(
        "Try a different input.", "A girl playing with her dog on the beach."
    )

    submit_button = st.button("Submit")

    col1, col2 = st.columns(2)

    if file:
        raw_img = Image.open(file).convert("RGB")
    else:
        raw_img = load_demo_image()

    col1.header("Image")
    w, h = raw_img.size
    scaling_factor = 720 / w
    resized_image = raw_img.resize((int(w * scaling_factor), int(h * scaling_factor)))
    col1.image(resized_image, use_column_width=True)

    col2.header("GradCam")

    if submit_button:
        if model_type.startswith("BLIP"):
            blip_type = model_type.split("_")[1]
            model = load_blip_itm_model(device, model_type=blip_type)

        img = vis_processor(raw_img).unsqueeze(0).to(device)
        qry = text_processor(query)

        qry_tok = tokenizer(qry, return_tensors="pt").to(device)

        norm_img = np.float32(resized_image) / 255

        gradcam, _ = compute_gradcam(model, img, qry, qry_tok, block_num=layer_num)

        avg_gradcam = getAttMap(norm_img, gradcam[0][1], blur=True)
        col2.image(avg_gradcam, use_column_width=True, clamp=True)

        num_cols = 4.0
        num_tokens = len(qry_tok.input_ids[0]) - 2

        num_rows = int(math.ceil(num_tokens / num_cols))

        gradcam_iter = iter(gradcam[0][2:-1])
        token_id_iter = iter(qry_tok.input_ids[0][1:-1])

        for _ in range(num_rows):
            with st.container():
                for col in st.columns(int(num_cols)):
                    token_id = next(token_id_iter, None)
                    if not token_id:
                        break
                    gradcam_img = next(gradcam_iter)

                    word = tokenizer.decode([token_id])
                    gradcam_todraw = getAttMap(norm_img, gradcam_img, blur=True)

                    new_title = (
                        '<p style="text-align: center; font-size: 25px;">{}</p>'.format(
                            word
                        )
                    )
                    col.markdown(new_title, unsafe_allow_html=True)
                    # st.image(image, channels="BGR")
                    col.image(gradcam_todraw, use_column_width=True, clamp=True)
