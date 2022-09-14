"""
 # Copyright (c) 2022, salesforce.com, inc.
 # All rights reserved.
 # SPDX-License-Identifier: BSD-3-Clause
 # For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import random
from collections import OrderedDict
from functools import reduce
from tkinter import N

import streamlit as st
from lavis.common.registry import registry
from lavis.datasets.builders import dataset_zoo, load_dataset
from lavis.datasets.builders.base_dataset_builder import load_dataset_config
from PIL import Image

IMAGE_LAYOUT = 3, 4
VIDEO_LAYOUT = 1, 2

PREV_STR = "Prev"
NEXT_STR = "Next"


def sample_dataset(dataset, indices):
    samples = [dataset.displ_item(idx) for idx in indices]

    return samples


def get_concat_v(im1, im2):
    margin = 5

    canvas_size = (im1.width + im2.width + margin, max(im1.height, im2.height))
    canvas = Image.new("RGB", canvas_size, "White")
    canvas.paste(im1, (0, 0))
    canvas.paste(im2, (im1.width + margin, 0))

    return canvas


def resize_img_w(raw_img, new_w=224):
    if isinstance(raw_img, list):
        resized_imgs = [resize_img_w(img, 196) for img in raw_img]
        # concatenate images
        resized_image = reduce(get_concat_v, resized_imgs)
    else:
        w, h = raw_img.size
        scaling_factor = new_w / w
        resized_image = raw_img.resize(
            (int(w * scaling_factor), int(h * scaling_factor))
        )

    return resized_image


def get_visual_key(dataset):
    if "image" in dataset[0]:
        return "image"
    elif "image0" in dataset[0]:  # NLVR2 dataset
        return "image"
    elif "video" in dataset[0]:
        return "video"
    else:
        raise ValueError("Visual key not found.")


def gather_items(samples, exclude=[]):
    gathered = []

    for s in samples:
        ns = OrderedDict()
        for k in s.keys():
            if k not in exclude:
                ns[k] = s[k]

        gathered.append(ns)

    return gathered


@st.cache(allow_output_mutation=True)
def load_dataset_cache(name):
    return load_dataset(name)


def format_text(text):
    md = "\n\n".join([f"**{k}**: {v}" for k, v in text.items()])

    return md


def show_samples(dataset, offset=0, is_next=False):
    visual_key = get_visual_key(dataset)

    num_rows, num_cols = IMAGE_LAYOUT if visual_key == "image" else VIDEO_LAYOUT
    n_samples = num_rows * num_cols

    if not shuffle:
        if is_next:
            start = min(int(start_idx) + offset + n_samples, len(dataset) - n_samples)
        else:
            start = max(0, int(start_idx) + offset - n_samples)

        st.session_state.last_start = start
        end = min(start + n_samples, len(dataset))

        indices = list(range(start, end))
    else:
        indices = random.sample(range(len(dataset)), n_samples)
    samples = sample_dataset(dataset, indices)

    visual_info = (
        iter([resize_img_w(s[visual_key]) for s in samples])
        if visual_key == "image"
        # else iter([s[visual_key] for s in samples])
        else iter([s["file"] for s in samples])
    )
    text_info = gather_items(samples, exclude=["image", "video"])
    text_info = iter([format_text(s) for s in text_info])

    st.markdown(
        """<hr style="height:1px;border:none;color:#c7ccd4;background-color:#c7ccd4;"/> """,
        unsafe_allow_html=True,
    )
    for _ in range(num_rows):
        with st.container():
            for col in st.columns(num_cols):
                # col.text(next(text_info))
                # col.caption(next(text_info))
                try:
                    col.markdown(next(text_info))
                    if visual_key == "image":
                        col.image(next(visual_info), use_column_width=True, clamp=True)
                    elif visual_key == "video":
                        col.markdown(
                            "![Alt Text](https://media.giphy.com/media/vFKqnCdLPNOKc/giphy.gif)"
                        )
                except StopIteration:
                    break

            st.markdown(
                """<hr style="height:1px;border:none;color:#c7ccd4;background-color:#c7ccd4;"/> """,
                unsafe_allow_html=True,
            )

    st.session_state.n_display = n_samples


if __name__ == "__main__":
    st.set_page_config(
        page_title="LAVIS Dataset Explorer",
        # layout="wide",
        initial_sidebar_state="expanded",
    )

    dataset_name = st.sidebar.selectbox("Dataset:", dataset_zoo.get_names())

    function = st.sidebar.selectbox("Function:", ["Browser"], index=0)

    if function == "Browser":
        shuffle = st.sidebar.selectbox("Shuffled:", [True, False], index=0)

        dataset = load_dataset_cache(dataset_name)
        split = st.sidebar.selectbox("Split:", dataset.keys())

        dataset_len = len(dataset[split])
        st.success(
            f"Loaded {dataset_name}/{split} with **{dataset_len}** records.  **Image/video directory**: {dataset[split].vis_root}"
        )

        if "last_dataset" not in st.session_state:
            st.session_state.last_dataset = dataset_name
            st.session_state.last_split = split

        if "last_start" not in st.session_state:
            st.session_state.last_start = 0

        if "start_idx" not in st.session_state:
            st.session_state.start_idx = 0

        if "shuffle" not in st.session_state:
            st.session_state.shuffle = shuffle

        if "first_run" not in st.session_state:
            st.session_state.first_run = True
        elif (
            st.session_state.last_dataset != dataset_name
            or st.session_state.last_split != split
        ):
            st.session_state.first_run = True

            st.session_state.last_dataset = dataset_name
            st.session_state.last_split = split
        elif st.session_state.shuffle != shuffle:
            st.session_state.shuffle = shuffle
            st.session_state.first_run = True

        if not shuffle:
            n_col, p_col = st.columns([0.05, 1])

            prev_button = n_col.button(PREV_STR)
            next_button = p_col.button(NEXT_STR)

        else:
            next_button = st.button(NEXT_STR)

        if not shuffle:
            start_idx = st.sidebar.text_input(f"Begin from (total {dataset_len})", 0)

            if not start_idx.isdigit():
                st.error(f"Input to 'Begin from' must be digits, found {start_idx}.")
            else:
                if int(start_idx) != st.session_state.start_idx:
                    st.session_state.start_idx = int(start_idx)
                    st.session_state.last_start = int(start_idx)

            if prev_button:
                show_samples(
                    dataset[split],
                    offset=st.session_state.last_start - st.session_state.start_idx,
                    is_next=False,
                )

        if next_button:
            show_samples(
                dataset[split],
                offset=st.session_state.last_start - st.session_state.start_idx,
                is_next=True,
            )

        if st.session_state.first_run:
            st.session_state.first_run = False

            show_samples(
                dataset[split],
                offset=st.session_state.last_start - st.session_state.start_idx,
                is_next=True,
            )
