import streamlit as st
from PIL import Image
import os, time, subprocess
from app import  pending_job_path, job_output_path
from app.utils import create_uniq_user_job_name

job_type='txt2image'

def compute_grid(num_images):
    cols = []
    for i in range(num_images):
        cols.append(st.columns())
    return cols

def app():
    num_images = st.sidebar.slider("Choose number of images to generate",
                                    min_value=1, max_value=3, step=1)
    random_seed = st.sidebar.text_input("Seed:", 1024,
                                        help="Set random seed to reproduce the generated images")
    st.markdown(
        "<h1 style='text-align: center;'>Image Generation</h1>", unsafe_allow_html=True
    )
    row1_1, row1_spacer1, row1_2, row1_spacer2 = st.columns((15.5, .1, 3.5, 0.1))
    with row1_1:
        user_prompt = st.text_input(
            "Describe the image you would like to generate",
            "a painting of Singapore Garden By the Bay in the style of Vincent Van Gogh",
            help="Try something creative."
        )
    with row1_2:
        st.write("")
        st.write("")
        generation_button = st.button("Generate")

    if generation_button:
        time_stamp = str(time.time())
        file_name = str(random_seed)+'\t'+str(time_stamp)+'\t'+user_prompt[:50]+'\t'+str(num_images)+'.txt'
        pending_path = os.path.join(pending_job_path, job_type)
        if not os.path.exists(pending_path):
            subprocess.run(['mkdir', '-p', pending_path], shell=False)
        with open(os.path.join(pending_path,file_name),'w') as new_job:
            line = str(random_seed)+'\t'+time_stamp+'\t'+user_prompt+'\t'+str(num_images)
            new_job.write(line+'\n')
            new_job.close()
        outpath = os.path.join(job_output_path, job_type)
        if not os.path.exists(outpath):
            os.mkdir(outpath)
        sample_path = os.path.join(outpath, "samples")
        if not os.path.exists(sample_path):
            subprocess.run(['mkdir', '-p', sample_path], shell=False)
        generated_image = sample_path+'/{}_grid.png'.format(create_uniq_user_job_name(time_stamp, user_prompt))
        num_pending_jobs = len(os.listdir(pending_path))
        with st.spinner("Queuing (#{} in line) for generation".format(num_pending_jobs)):
            while True:
                if os.path.exists(generated_image):
                    time.sleep(1)
                    cols = st.columns(num_images)
                    for i in range(1, num_images+1):
                        img = sample_path+'/{}_{}.png'.format(create_uniq_user_job_name(time_stamp, user_prompt), i)
                        with cols[i-1]:
                            st.image(Image.open(img))
                    break
