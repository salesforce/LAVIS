import requests
import streamlit as st
from app import device
from lavis.models import load_model_and_preprocess
from PIL import Image


@st.cache()
def load_demo_image():
    response = requests.get(
        'https://storage.googleapis.com/sfr-vision-language-research/LAVIS/assets/merlion.png',  # noqa
        stream=True).raw
    img = Image.open(response).convert('RGB')
    return img


@st.experimental_singleton
def get_model():
    model, vis_processors, _ = load_model_and_preprocess(
        name='blip2_t5',
        model_type='pretrain_flant5xxl',
        is_eval=True,
        device=device)
    return model, vis_processors


def app():

    st.markdown(
        "<h1 style='text-align: center;'>BLIP2 Instructed Zero-Shot Image-to-text Generation</h1>",  # noqa
        unsafe_allow_html=True,
    )

    model, vis_processors = get_model()

    file = st.file_uploader('Try the provided image or upload your own:')

    container1 = st.container()
    if file:
        raw_img = Image.open(file).convert('RGB')
    else:
        raw_img = load_demo_image()

    container1.header('Image')

    w, h = raw_img.size
    scaling_factor = 360 / w
    resized_image = raw_img.resize(
        (int(w * scaling_factor), int(h * scaling_factor)))

    container1.image(resized_image, use_column_width='auto')

    # save session history
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    question = st.text_input('Questions', 'which city is this?')
    button = st.button('Chat')

    # search on both selection of topk and button
    if button:
        image = vis_processors['eval'](raw_img).unsqueeze(0).to(device)
        template = 'Question: {} Answer: {}.'
        prompt = ' '.join([
            template.format(pair[0], pair[1])
            for pair in st.session_state['history']
        ]) + ' Question: ' + question + ' Answer:'

        output = model.generate({'image': image, 'prompt': prompt})
        answer = output[0]
        st.session_state['history'].append((question, answer))
        st.write(answer)

        st.markdown('Chat History:', )
        hist_temp = 'Your: {} <br /> BLIP2: {} <br /> '
        history = ''.join([
            hist_temp.format(pair[0], pair[1])
            for pair in st.session_state['history']
        ])
        st.write(history, unsafe_allow_html=True)

    button2 = st.button('Empty history')
    if button2:
        st.session_state['history'] = []
