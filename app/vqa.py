import streamlit as st
from app import load_demo_image, device
from app.utils import load_model_cache
from lavis.processors import load_processor
from PIL import Image


def app():
    model_type = st.sidebar.selectbox("Model:", ["BLIP", "BLIP_aokvqa", "BLIP_okvqa"])

    # ===== layout =====
    st.markdown(
        "<h1 style='text-align: center;'>Visual Question Answering</h1>",
        unsafe_allow_html=True,
    )

    instructions = """Try the provided image or upload your own:"""
    file = st.file_uploader(instructions)

    col1, col2 = st.columns(2)

    col1.header("Image")
    if file:
        raw_img = Image.open(file).convert("RGB")
    else:
        raw_img = load_demo_image()

    w, h = raw_img.size
    scaling_factor = 720 / w
    resized_image = raw_img.resize((int(w * scaling_factor), int(h * scaling_factor)))

    col1.image(resized_image, use_column_width=True)
    col2.header("Question")

    with col2:
        #user_question = col2.text_input("Input your question!", "What are objects there?")
        #question = '<p style="font-family:sans-serif; color:Black; font-size: 25px;">Input your question!</p>'
        #st.markdown(question, unsafe_allow_html=True)
        user_question = col2.text_area("Input your question!", "What are objects there?")
        qa_button = st.button("Answer my question")
    
    #col2.header("Answer")

    # ===== event =====
    vis_processor = load_processor("blip_image_eval").build(image_size=480)
    text_processor = load_processor("blip_question").build()

    if qa_button:
        if model_type.startswith("BLIP"):
            model = load_model_cache(
                "blip_vqa", model_type="vqav2", is_eval=True, device=device
            )
            if model_type == "BLIP_aokvqa":
                model.load_from_pretrained("https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP/blip_aokvqa.pth")
            elif model_type == "BLIP_okvqa":
                model.load_from_pretrained("https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP/blip_okvqa.pth")
            img = vis_processor(raw_img).unsqueeze(0).to(device)
            question = text_processor(user_question)

            vqa_samples = {"image": img, "text_input": [question]}
            answers = model.predict_answers(vqa_samples, inference_method="generate")

            with col2:
                st.header("Answer")
                for answer in answers: 
                    answer_md = '<p style="font-family:sans-serif; color:Black; font-size: 25px;">{}</p>'.format(answer)
                    st.markdown(answer_md, unsafe_allow_html=True)
            #col2.write("\n".join(answers), use_column_width=True)
