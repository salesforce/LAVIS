from PIL import Image
import requests

import streamlit as st
import torch


@st.cache()
def load_demo_image():
    img_url = (
        "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"
    )
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
    return raw_image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cache_root = "/export/home/.cache/lavis/"
pending_job_path = "app/task_queues/pending_jobs/"
finished_job_path = "app/task_queues/finished_jobs/"
job_output_path = "app/task_queues/outputs/"
