import os, time, subprocess

import numpy as np
import streamlit as st
import torch
from app import cache_root, device, pending_job_path, job_output_path
from app.utils import create_uniq_user_job_name


job_type = 'search'

def app():
    # === layout ===
    model_type = st.sidebar.selectbox("Model:", ["BLIP_base", "BLIP_large"])
    if model_type.startswith("BLIP"):
        blip_type = model_type.split("_")[1]
    file_root = os.path.join(cache_root, "coco/images/train2014/")

    values = [16, 24, 48]
    default_layer_num = values.index(24)
    num_display = st.sidebar.selectbox(
        "Number of images:", values, index=default_layer_num
    )
    show_gradcam = st.sidebar.selectbox("Show GradCam:", [True, False], index=1)
    itm_ranking = st.sidebar.selectbox("Multimodal re-ranking:", [True, False], index=0)

    st.markdown(
        "<h1 style='text-align: center;'>Multimodal Search</h1>", unsafe_allow_html=True
    )

    row1_1, row1_spacer1, row1_2, row1_spacer2 = st.columns((15.5, .1, 3.5, 0.1))
    with row1_1:
        user_question = st.text_input(
            "Search query", "A dog running on the grass.", help="Type something to search."
        )
    with row1_2:
        st.markdown("")
        st.markdown("")
        search_button = st.button("Search")

    if search_button:
        time_stamp = time.time()
        pending_path = os.path.join(pending_job_path, job_type)
        if not os.path.exists(pending_path):
            subprocess.run(['mkdir', '-p', pending_path], shell=False)
        file_name = '{}_result.txt'.format(create_uniq_user_job_name(str(time_stamp), user_question))
        with open(os.path.join(pending_path, file_name),'w') as new_job:
            line = str(time_stamp)+'\t'+user_question+'\t'+str(num_display)+'\t'+blip_type
            new_job.write(line+'\n')
            new_job.close()

        num_pending_jobs = len(os.listdir(pending_path))
        outpath = os.path.join(job_output_path, job_type)
        search_result = os.path.join(outpath, file_name)

        filenames = []
        with st.spinner("Queuing (#{} in line)".format(num_pending_jobs)):
            while True:
                if os.path.exists(search_result):
                    time.sleep(1)
                    with open(search_result) as f:
                        count = 0
                        for line in f:
                            if count < num_display:
                                p = os.path.join(file_root, line.rstrip('\n'))
                                filenames.append(p)
                                count += 1
                    break
    # ========= ITM and GradCam ==========

        itm_scores_pt = outpath+'/{}_itm.pt'.format(create_uniq_user_job_name(str(time_stamp), user_question))
        itm_scores = torch.load(itm_scores_pt, map_location=torch.device('cpu'))
        os.remove(itm_scores_pt)

        avg_gradcams_pt = outpath+'/{}_avg_gradcams.npy'.format(create_uniq_user_job_name(str(time_stamp), user_question))
        avg_gradcams = np.load(avg_gradcams_pt, allow_pickle=True)

        os.remove(avg_gradcams_pt)

        all_raw_images_pt = outpath+'/{}_all_raw_images.npy'.format(create_uniq_user_job_name(str(time_stamp), user_question))
        all_raw_images = np.load(all_raw_images_pt, allow_pickle=True)
        os.remove(all_raw_images_pt)

        # ========= ITM re-ranking =========
        if itm_ranking:
            itm_scores_sorted, indices = torch.sort(itm_scores, descending=True)

            avg_gradcams_sorted = []
            all_raw_images_sorted = []
            for idx in indices:
                avg_gradcams_sorted.append(avg_gradcams[idx])
                all_raw_images_sorted.append(all_raw_images[idx])

            avg_gradcams = avg_gradcams_sorted
            all_raw_images = all_raw_images_sorted

        if show_gradcam:
            images_to_show = iter(avg_gradcams)
        else:
            images_to_show = iter(all_raw_images)

        num_cols = 4
        num_rows = int(num_display / num_cols)
        for _ in range(num_rows):
            with st.container():
                for col in st.columns(num_cols):
                    col.image(next(images_to_show), use_column_width=True, clamp=True)





