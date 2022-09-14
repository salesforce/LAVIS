"""
 # Copyright (c) 2022, salesforce.com, inc.
 # All rights reserved.
 # SPDX-License-Identifier: BSD-3-Clause
 # For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from app.multipage import MultiPage
from app import vqa, caption
from app import image_text_match as itm
from app import text_localization as tl
from app import multimodal_search as ms
from app import classification as cl


if __name__ == "__main__":
    app = MultiPage()

    app.add_page("Image Description Generation", caption.app)
    app.add_page("Multimodal Search", ms.app)
    app.add_page("Visual Question Answering", vqa.app)
    app.add_page("Image Text Matching", itm.app)
    app.add_page("Text Localization", tl.app)
    app.add_page("Classification", cl.app)
    app.run()
