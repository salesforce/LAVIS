from app.multipage import MultiPage
from app import vqa, caption
from app import image_text_match as itm
from app import text_localization as tl
from app import multimodal_search as ms


if __name__ == "__main__":
    app = MultiPage()

    app.add_page("Multimodal Search", ms.app)
    app.add_page("Image Description Generation", caption.app)
    app.add_page("Visual Question Answering", vqa.app)
    app.add_page("Image Text Matching", itm.app)
    app.add_page("Text Localization", tl.app)
    app.run()
