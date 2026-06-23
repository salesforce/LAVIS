from app.multipage import MultiPage
from app import vqa, caption
#from app import caption_front_end as caption
from app import image_text_match as itm
from app import text_localization as tl
#from app import multimodal_search as ms
from app import multimodal_search_front_end as ms
from app import classification as cl
from PIL import Image
import streamlit as st
from app import txt2image_front_end as ig


if __name__ == "__main__":
    app = MultiPage()

    logo = Image.open("app/logo_color.png")
    st.sidebar.image(logo.resize((592, 157)))

    # add Salesforce Logo on top right
    st.markdown(
        "<img id='sf-logo' src='https://c1.sfdcstatic.com/content/dam/sfdc-docs/www/logos/logo-salesforce.svg' style='position: absolute;top: 25px;right: 5px;'></img>",
        unsafe_allow_html=True,
    )

    # load custom css
    with open("app/style_override.css") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    app.add_page("Image Description Generation", caption.app)
    app.add_page("Multimodal Search", ms.app)
    app.add_page("Visual Question Answering", vqa.app)
    app.add_page("Zero-shot Image Classification", cl.app)
    app.add_page("Text-to-Image Generation", ig.app)
    app.add_page("Text Visualization", tl.app)
    app.run()
