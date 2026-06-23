import streamlit as st
from app import device, load_demo_image
from app.utils import load_model_cache
from lavis.processors import load_processor
from PIL import Image

import random
import numpy as np
import torch


def setup_seeds(seed):
    random.seed(seed)
    np.random.seed(int(seed))
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = False
        cudnn.deterministic = True


def app():
    # ===== layout =====
    model_type = st.sidebar.selectbox("Model:", ["BLIP_large", "BLIP_base"])

    sampling_method = st.sidebar.selectbox(
        "Sampling method:", ["Beam search", "Nucleus sampling"]
    )
    num_captions = 1
    if sampling_method == "Nucleus sampling":
        random_seed = st.sidebar.text_input("Seed:", 1024, help="Set random seed to reproduce the image description")
        setup_seeds(random_seed)
        num_captions = st.sidebar.slider("Choose number of captions to generate",
                                    min_value=1, max_value=5, step=1)

    st.markdown(
        "<h1 style='text-align: center;'>Image Description Generation</h1>",
        unsafe_allow_html=True,
    )

    instructions = """Try the provided image or upload your own:"""
    file = st.file_uploader(instructions)

    use_beam = sampling_method == "Beam search"

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
    #col2.header("Description")
    #with col2:
    cap_button = st.button("Generate")

    # ==== event ====
    vis_processor = load_processor("blip_image_eval").build(image_size=384)

    if cap_button:
        if model_type.startswith("BLIP"):
            blip_type = model_type.split("_")[1].lower()
            model = load_model_cache(
                "blip_caption",
                model_type=f"{blip_type}_coco",
                is_eval=True,
                device=device,
            )

        img = vis_processor(raw_img).unsqueeze(0).to(device)
        captions = generate_caption(
            model=model, image=img, use_nucleus_sampling=not use_beam, num_captions=num_captions
        )
 
        #with col2:
        #    for caption in captions:
        #        caption_md = '<p style="font-family:sans-serif; color:Black; font-size: 20px;">{}</p>'.format(caption)
        #        st.markdown(caption_md, unsafe_allow_html=True)
        with col2:
            st.header("Description")
        #with col2:
            for caption in captions:
                caption_md = '<p style="font-family:sans-serif; color:Black; font-size: 25px;">{}</p>'.format(caption)
                st.markdown(caption_md, unsafe_allow_html=True)
        #col2.write("\n\n".join(captions), use_column_width=True)


def generate_caption(
    model, image, use_nucleus_sampling=False, num_captions = 1, num_beams=3, max_length=40, min_length=5
):
    samples = {"image": image}

    captions = []
    if use_nucleus_sampling:
        #for _ in range(5):
        captions = model.generate(
                samples,
                use_nucleus_sampling=True,
                max_length=max_length,
                min_length=min_length,
                top_p=0.9,
                num_captions=num_captions
        )
        #captions.append(caption[0])
    else:
        caption = model.generate(
            samples,
            use_nucleus_sampling=False,
            num_beams=num_beams,
            max_length=max_length,
            min_length=min_length,
            num_captions=1
        )
        captions.append(caption[0])

    return captions
