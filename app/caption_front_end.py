import streamlit as st
from app import load_demo_image, job_output_path, pending_job_path
from app.utils import create_uniq_user_job_name
from lavis.processors import load_processor
from PIL import Image

import os, time, subprocess
import random
import numpy as np
import torch

job_type = 'caption'

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

    vis_processor = load_processor("blip_image_eval").build(image_size=384)
    img = vis_processor(raw_img).unsqueeze(0)

    col2.header("Description")
    #with col2:
    cap_button = st.button("Generate")
    blip_type = model_type.split("_")[1].lower()

    if cap_button:
        time_stamp = time.time()
        pending_jobs = os.path.join(pending_job_path, job_type)
        if not os.path.exists(pending_jobs):
            #os.makedirs(pending_jobs)
            subprocess.run(['mkdir', '-p', pending_jobs], shell=False)
        file_name = '{}_result.txt'.format(create_uniq_user_job_name(time_stamp, sampling_method))
        with open(os.path.join(pending_jobs, file_name),'w') as new_job:
            line = str(time_stamp)+'\t'+blip_type+'\t'+str(sampling_method)+'\t'+str(num_captions)
            new_job.write(line+'\n')
            new_job.close()

        num_pending_jobs = len(os.listdir(pending_jobs))
        outpath = os.path.join(job_output_path,job_type)
        if not os.path.exists(outpath):
            subprocess.run(['mkdir', '-p', outpath], shell=False)
        search_result = outpath+'/{}_result.txt'.format(create_uniq_user_job_name(time_stamp, sampling_method))
        torch.save(img, outpath+'/{}_raw_image.pt'.format(create_uniq_user_job_name(time_stamp, sampling_method)))

        with st.spinner("Queuing (#{} in line)".format(num_pending_jobs)):
            while True:
                if os.path.exists(search_result):
                    time.sleep(1)
                    with open(search_result) as f:
                        count = 0
                        with col2:
                            #st.header("Description")
                            for caption in f:
                                caption = caption.rstrip(' \n')
                                if count < num_captions:
                                    caption_md = '<p style="font-family:sans-serif; color:Black; font-size: 25px;">{}</p>'.format(caption)
                                    st.markdown(caption_md, unsafe_allow_html=True)
                                    count += 1
                    break
