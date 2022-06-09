import math
from PIL import Image
import os
import requests

import streamlit as st
import numpy as np
import torch

import os, sys

sys.path.append(".")

# from lavis.common.registry import registry
# from lavis.processors import *
# from lavis.models import *

# from lavis.models.blip_models import init_tokenizer

from lavis.common.registry import registry
from lavis.processors import *
from lavis.models import *

from lavis.models.blip_models import init_tokenizer


def read_img(filepath):
    raw_image = Image.open(filepath).convert("RGB")

    return raw_image


@st.cache(
    hash_funcs={
        torch.nn.parameter.Parameter: lambda parameter: parameter.data.detach()
        .cpu()
        .numpy()
    },
    allow_output_mutation=True,
)
def load_blip_caption_model(device, model_type="base"):
    model = registry.get_model_class("blip_caption").build_default_model(
        model_type=model_type
    )
    model.eval()
    model = model.to(device)
    return model


@st.cache(
    hash_funcs={
        torch.nn.parameter.Parameter: lambda parameter: parameter.data.detach()
        .cpu()
        .numpy()
    },
    allow_output_mutation=True,
)
def load_blip_vqa_model(device, model_type="base"):
    model = registry.get_model_class("blip_vqa").build_default_model(
        model_type=model_type
    )
    model.eval()
    model = model.to(device)
    return model


@st.cache(
    hash_funcs={
        torch.nn.parameter.Parameter: lambda parameter: parameter.data.detach()
        .cpu()
        .numpy()
    },
    allow_output_mutation=True,
)
def load_blip_itm_model(device, model_type="base"):
    if model_type == "large":
        pretrained_path = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_retrieval_coco.pth"
    else:
        pretrained_path = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_retrieval_coco.pth"
    model = BlipITM(pretrained=pretrained_path, vit=model_type)
    model.eval()
    model = model.to(device)
    return model


@st.cache(
    hash_funcs={
        torch.nn.parameter.Parameter: lambda parameter: parameter.data.detach()
        .cpu()
        .numpy()
    },
    allow_output_mutation=True,
)
def load_feature_extractor_model(device):
    model_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base.pth"
    model = BlipBase(pretrained=model_url)

    model.eval()
    model = model.to(device)
    return model


@st.cache(
    hash_funcs={
        torch.nn.parameter.Parameter: lambda parameter: parameter.data.detach()
        .cpu()
        .numpy()
    },
    allow_output_mutation=True,
)
def load_feat():
    path2feat = torch.load(
        os.path.join(
            os.path.dirname(__file__), "resources/path2feat_coco_train2014.pth"
        )
    )
    paths = sorted(path2feat.keys())

    all_img_feats = torch.stack([path2feat[k] for k in paths], dim=0).to(device)

    return path2feat, paths, all_img_feats


@st.cache(allow_output_mutation=True)
def init_bert_tokenizer():
    tokenizer = init_tokenizer()
    return tokenizer


@st.cache()
def load_demo_image():
    img_url = (
        "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"
    )
    # img_url = "/export/home/workspace/LAVIS/lavis/app/resources/blue-blouse.jpg"
    # img_url = "/export/home/workspace/LAVIS/lavis/app/resources/marc.jpeg"
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
    # raw_image = Image.open(img_url).convert("RGB")

    w, h = raw_image.size
    # display(raw_image.resize((w//3,h//3)))

    return raw_image


def generate_caption(
    model, image, use_nucleus_sampling=False, num_beams=3, max_length=40, min_length=5
):
    samples = {"image": image}

    captions = []
    if use_nucleus_sampling:
        for _ in range(5):
            caption = model.generate(
                samples,
                use_nucleus_sampling=True,
                max_length=max_length,
                min_length=min_length,
                top_p=0.9,
            )
            captions.append(caption[0])
    else:
        caption = model.generate(
            samples,
            use_nucleus_sampling=False,
            num_beams=num_beams,
            max_length=max_length,
            min_length=min_length,
        )
        captions.append(caption[0])

    return captions


from skimage import transform as skimage_transform
from scipy.ndimage import filters
from matplotlib import pyplot as plt


def getAttMap(img, attMap, blur=True, overlap=True):
    attMap -= attMap.min()
    if attMap.max() > 0:
        attMap /= attMap.max()
    attMap = skimage_transform.resize(attMap, (img.shape[:2]), order=3, mode="constant")
    if blur:
        attMap = filters.gaussian_filter(attMap, 0.02 * max(img.shape[:2]))
        attMap -= attMap.min()
        attMap /= attMap.max()
    cmap = plt.get_cmap("jet")
    attMapV = cmap(attMap)
    attMapV = np.delete(attMapV, 3, 2)
    if overlap:
        attMap = (
            1 * (1 - attMap**0.7).reshape(attMap.shape + (1,)) * img
            + (attMap**0.7).reshape(attMap.shape + (1,)) * attMapV
        )
    return attMap


def compute_gradcam(model, visual_input, text_input, tokenized_text, block_num=6):
    model.text_encoder.base_model.base_model.encoder.layer[
        block_num
    ].crossattention.self.save_attention = True

    output = model(visual_input, text_input, match_head="itm")
    loss = output[:, 1].sum()

    model.zero_grad()
    loss.backward()
    with torch.no_grad():
        mask = tokenized_text.attention_mask.view(
            tokenized_text.attention_mask.size(0), 1, -1, 1, 1
        )  # (bsz,1,token_len, 1,1)
        token_length = mask.sum() - 2
        token_length = token_length.cpu()
        # grads and cams [bsz, num_head, seq_len, image_patch]
        grads = model.text_encoder.base_model.base_model.encoder.layer[
            block_num
        ].crossattention.self.get_attn_gradients()
        cams = model.text_encoder.base_model.base_model.encoder.layer[
            block_num
        ].crossattention.self.get_attention_map()

        # assume using vit large with 576 num image patch
        cams = cams[:, :, :, 1:].reshape(visual_input.size(0), 12, -1, 24, 24) * mask
        grads = (
            grads[:, :, :, 1:].clamp(0).reshape(visual_input.size(0), 12, -1, 24, 24)
            * mask
        )

        gradcam = cams * grads
        gradcam = gradcam[0].mean(0).cpu().detach()
        # [enc token gradcam, average gradcam across token, gradcam for individual token]
        gradcam = torch.cat(
            (
                gradcam[0:1, :],
                gradcam[1 : token_length + 1, :].sum(dim=0, keepdim=True)
                / token_length,
                gradcam[1:, :],
            )
        )

    return gradcam, output


def show_img_caption():
    sampling_method = st.sidebar.selectbox(
        "Sampling method:", ["Beam search", "Nucleus sampling"]
    )

    vis_processor = BlipImageEvalProcessor(image_size=384)
    # st.markdown(
    #     "<h1 style='text-align: center;'>Image Captioning</h1>", unsafe_allow_html=True
    # )
    # st.markdown(
    #     "<h1 style='text-align: center;'>Product Description Generation</h1>",
    #     unsafe_allow_html=True,
    # )
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

    # caption_button = st.button("Submit")

    # col1.header("Product image")
    col1.header("Image")

    w, h = raw_img.size
    scaling_factor = 720 / w
    resized_image = raw_img.resize((int(w * scaling_factor), int(h * scaling_factor)))

    col1.image(resized_image, use_column_width=True)
    # col2.header("Captions")
    col2.header("Description")

    cap_button = st.button("Generate")

    if cap_button:
        if model_type.startswith("BLIP"):
            blip_type = model_type.split("_")[1]
            model = load_blip_caption_model(device, model_type=blip_type)

        img = vis_processor(raw_img).unsqueeze(0).to(device)
        captions = generate_caption(
            model=model, image=img, use_nucleus_sampling=not use_beam
        )

        col2.write("\n\n".join(captions), use_column_width=True)


def show_vqa():
    vis_processor = BlipImageEvalProcessor(image_size=480)

    st.markdown(
        "<h1 style='text-align: center;'>Visual Question Answering</h1>",
        unsafe_allow_html=True,
    )

    instructions = """Try the provided image or upload your own:"""
    file = st.file_uploader(instructions)

    col1, col2 = st.columns(2)

    # st.header("Image")
    col1.header("Image")
    if file:
        raw_img = Image.open(file).convert("RGB")
    else:
        raw_img = load_demo_image()

    w, h = raw_img.size
    scaling_factor = 720 / w
    resized_image = raw_img.resize((int(w * scaling_factor), int(h * scaling_factor)))
    # st.image(resized_image, use_column_width=True)
    col1.image(resized_image, use_column_width=True)

    # st.header("Question")
    col2.header("Question")
    # user_question = st.text_input(
    user_question = col2.text_input("Input your question!", "What are objects there?")
    # user_question = col2.text_input("Input your question here.", "")
    qa_button = st.button("Submit")

    # col_cam, col_ans = st.columns(2)
    # col_cam.header("GradCam")
    # st.header("Answer")
    col2.header("Answer")

    if qa_button:
        if model_type.startswith("BLIP"):
            blip_type = model_type.split("_")[1]
            if blip_type == "large":
                st.warning("No large model provided.")
            else:
                model = load_blip_vqa_model(device, model_type=blip_type)

                img = vis_processor(raw_img).unsqueeze(0).to(device)
                question = text_processor(user_question)

                vqa_samples = {"image": img, "text_input": [question]}
                answers = model.predict_answers(
                    vqa_samples, inference_method="generate"
                )

                # st.write("\n".join(answers), use_column_width=True)
                col2.write("\n".join(answers), use_column_width=True)


def compute_gradcam_batch(model, visual_input, text_input, tokenized_text, block_num=6):
    model.text_encoder.base_model.base_model.encoder.layer[
        block_num
    ].crossattention.self.save_attention = True

    output = model(visual_input, text_input, match_head="itm")
    loss = output[:, 1].sum()

    model.zero_grad()
    loss.backward()
    with torch.no_grad():
        mask = tokenized_text.attention_mask.view(
            tokenized_text.attention_mask.size(0), 1, -1, 1, 1
        )  # (bsz,1,token_len, 1,1)
        token_length = mask.sum() - 2
        token_length = token_length.cpu()
        # grads and cams [bsz, num_head, seq_len, image_patch]
        grads = model.text_encoder.base_model.base_model.encoder.layer[
            block_num
        ].crossattention.self.get_attn_gradients()
        cams = model.text_encoder.base_model.base_model.encoder.layer[
            block_num
        ].crossattention.self.get_attention_map()

        # assume using vit large with 576 num image patch
        cams = cams[:, :, :, 1:].reshape(visual_input.size(0), 12, -1, 24, 24) * mask
        grads = (
            grads[:, :, :, 1:].clamp(0).reshape(visual_input.size(0), 12, -1, 24, 24)
            * mask
        )

        gradcam = cams * grads
        # [enc token gradcam, average gradcam across token, gradcam for individual token]
        # gradcam = torch.cat((gradcam[0:1,:], gradcam[1:token_length+1, :].sum(dim=0, keepdim=True)/token_length, gradcam[1:, :]))
        gradcam = gradcam.mean(1).cpu().detach()
        gradcam = (
            gradcam[:, 1 : token_length + 1, :].sum(dim=1, keepdim=True) / token_length
        )

    return gradcam, output


def read_and_process_images(image_paths, vis_processor):
    raw_images = [read_img(path) for path in image_paths]
    images = [vis_processor(r_img) for r_img in raw_images]
    images_tensors = torch.stack(images).to(device)

    return raw_images, images_tensors


def resize_img(raw_img):
    w, h = raw_img.size
    scaling_factor = 240 / w
    resized_image = raw_img.resize((int(w * scaling_factor), int(h * scaling_factor)))
    return resized_image


def show_multimodal_search():
    file_root = "/export/home/.cache/lavis/coco/images/train2014/"

    values = [12, 24, 48]
    default_layer_num = values.index(24)
    num_display = st.sidebar.selectbox(
        "Number of images:", values, index=default_layer_num
    )
    show_gradcam = st.sidebar.selectbox("Show GradCam:", [True, False], index=1)
    itm_ranking = st.sidebar.selectbox("Multimodal re-ranking:", [True, False], index=0)

    vis_processor = BlipImageEvalProcessor(image_size=384)
    feature_extractor = load_feature_extractor_model(device)

    # st.title('Multimodal Search')
    st.markdown(
        "<h1 style='text-align: center;'>Multimodal Search</h1>", unsafe_allow_html=True
    )
    user_question = st.text_input(
        "Search query", "A dog running on the grass.", help="Type something to search."
    )
    user_question = text_processor(user_question)

    # ======= ITC =========
    with torch.no_grad():
        text_feature = feature_extractor(
            torch.zeros(0), user_question, mode="text", normalized=True
        )[0, 0]

        path2feat, paths, all_img_feats = load_feat()
        all_img_feats.to(device)

        num_cols = 4
        num_rows = int(num_display / num_cols)

        similarities = text_feature @ all_img_feats.T
        indices = torch.argsort(similarities, descending=True)[:num_display]

    top_paths = [paths[ind.detach().cpu().item()] for ind in indices]
    sorted_similarities = [similarities[idx] for idx in indices]
    filenames = [os.path.join(file_root, p) for p in top_paths]

    # ========= ITM and GradCam ==========
    bsz = 12  # max number of images to avoid cuda oom
    if model_type.startswith("BLIP"):
        blip_type = model_type.split("_")[1]

    itm_model = load_blip_itm_model(device, model_type=blip_type)

    tokenizer = init_bert_tokenizer()
    queries_batch = [user_question] * bsz
    queries_tok_batch = tokenizer(queries_batch, return_tensors="pt").to(device)

    num_batches = int(num_display / bsz)

    avg_gradcams = []
    all_raw_images = []
    itm_scores = []

    for i in range(num_batches):
        filenames_in_batch = filenames[i * bsz : (i + 1) * bsz]
        raw_images, images = read_and_process_images(filenames_in_batch, vis_processor)
        gradcam, itm_output = compute_gradcam_batch(
            itm_model, images, queries_batch, queries_tok_batch
        )

        all_raw_images.extend([resize_img(r_img) for r_img in raw_images])
        norm_imgs = [np.float32(r_img) / 255 for r_img in raw_images]

        for norm_img, grad_cam in zip(norm_imgs, gradcam):
            avg_gradcam = getAttMap(norm_img, grad_cam[0], blur=True)
            avg_gradcams.append(avg_gradcam)

        with torch.no_grad():
            itm_score = torch.nn.functional.softmax(itm_output, dim=1)

        itm_scores.append(itm_score)

    # ========= ITM re-ranking =========
    itm_scores = torch.cat(itm_scores)[:, 1]
    if itm_ranking:
        itm_scores_sorted, indices = torch.sort(itm_scores, descending=True)

        avg_gradcams_sorted = []
        all_raw_images_sorted = []
        for idx in indices:
            avg_gradcams_sorted.append(avg_gradcams[idx])
            all_raw_images_sorted.append(all_raw_images[idx])

        avg_gradcams = avg_gradcams_sorted
        all_raw_images = all_raw_images_sorted

        # scores_to_show = iter(itm_scores_sorted)
    # else:
    # scores_to_show = iter(itm_scores)

    if show_gradcam:
        images_to_show = iter(avg_gradcams)
    else:
        images_to_show = iter(all_raw_images)

    for _ in range(num_rows):
        with st.container():
            for col in st.columns(num_cols):
                # col.markdown("{:.3f}".format(next(scores_to_show).item()))
                col.image(next(images_to_show), use_column_width=True, clamp=True)


def show_text_localization():
    values = list(range(1, 12))
    default_layer_num = values.index(7)
    layer_num = (
        st.sidebar.selectbox("Layer number", values, index=default_layer_num) - 1
    )

    st.markdown(
        "<h1 style='text-align: center;'>Text Localization</h1>", unsafe_allow_html=True
    )

    vis_processor = BlipImageEvalProcessor(image_size=384)
    text_processor = BlipCaptionProcessor()

    tokenizer = init_bert_tokenizer()

    instructions = "Try the provided image and text or use your own ones."
    file = st.file_uploader(instructions)

    query = st.text_input(
        "Try a different input.", "A girl playing with her dog on the beach."
    )

    submit_button = st.button("Submit")

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

    col2.header("GradCam")

    if submit_button:
        if model_type.startswith("BLIP"):
            blip_type = model_type.split("_")[1]
            model = load_blip_itm_model(device, model_type=blip_type)

        img = vis_processor(raw_img).unsqueeze(0).to(device)
        qry = text_processor(query)

        qry_tok = tokenizer(qry, return_tensors="pt").to(device)

        norm_img = np.float32(resized_image) / 255

        gradcam, _ = compute_gradcam(model, img, qry, qry_tok, block_num=layer_num)

        avg_gradcam = getAttMap(norm_img, gradcam[1], blur=True)
        col2.image(avg_gradcam, use_column_width=True, clamp=True)

        num_cols = 4.0
        num_tokens = len(qry_tok.input_ids[0]) - 2

        num_rows = int(math.ceil(num_tokens / num_cols))

        gradcam_iter = iter(gradcam[2:-1])
        token_id_iter = iter(qry_tok.input_ids[0][1:-1])

        for _ in range(num_rows):
            with st.container():
                for col in st.columns(int(num_cols)):
                    token_id = next(token_id_iter, None)
                    if not token_id:
                        break
                    gradcam_img = next(gradcam_iter)

                    word = tokenizer.decode([token_id])
                    gradcam_todraw = getAttMap(norm_img, gradcam_img, blur=True)

                    new_title = (
                        '<p style="text-align: center; font-size: 25px;">{}</p>'.format(
                            word
                        )
                    )
                    col.markdown(new_title, unsafe_allow_html=True)
                    # st.image(image, channels="BGR")
                    col.image(gradcam_todraw, use_column_width=True, clamp=True)


def show_itm():
    if model_type.startswith("BLIP"):
        blip_type = model_type.split("_")[1]
        model = load_blip_itm_model(device, model_type=blip_type)

    vis_processor = BlipImageEvalProcessor(image_size=384)

    st.markdown(
        "<h1 style='text-align: center;'>Image Text Matching</h1>",
        unsafe_allow_html=True,
    )

    values = list(range(1, 12))
    default_layer_num = values.index(7)
    layer_num = (
        st.sidebar.selectbox("Layer number", values, index=default_layer_num) - 1
    )

    instructions = """Try the provided image or upload your own:"""
    file = st.file_uploader(instructions)

    col1, col2 = st.columns(2)
    col1.header("Image")
    col2.header("GradCam")
    if file:
        raw_img = Image.open(file).convert("RGB")
    else:
        raw_img = load_demo_image()

    w, h = raw_img.size
    scaling_factor = 720 / w
    resized_image = raw_img.resize((int(w * scaling_factor), int(h * scaling_factor)))
    col1.image(resized_image, use_column_width=True)

    col3, col4 = st.columns(2)
    col3.header("Text")
    user_question = col3.text_input(
        "Input your sentence!", "a woman sitting on the beach with a dog"
    )
    submit_button = col3.button("Submit")

    col4.header("Matching score")

    if submit_button:
        tokenizer = init_bert_tokenizer()

        img = vis_processor(raw_img).unsqueeze(0).to(device)
        qry = text_processor(user_question)

        norm_img = np.float32(resized_image) / 255

        qry_tok = tokenizer(qry, return_tensors="pt").to(device)
        gradcam, output = compute_gradcam(model, img, qry, qry_tok, block_num=layer_num)

        avg_gradcam = getAttMap(norm_img, gradcam[1], blur=True)

        col2.image(avg_gradcam, use_column_width=True, clamp=True)
        # output = model(img, question)
        itm_score = torch.nn.functional.softmax(output, dim=1)
        new_title = (
            '<p style="text-align: left; font-size: 25px;">\n{:.3f}%</p>'.format(
                itm_score[0][1].item() * 100
            )
        )
        # col4.markdown('<p style="text-align: center; font-size: 25px;">{:.3f}%</p>'.format(itm_score[0][1].item() * 100))
        col4.markdown(new_title, unsafe_allow_html=True)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text_processor = BlipCaptionProcessor()

    with st.sidebar.container():
        # st.sidebar.image(load_demo_image(), use_column_width=True)
        page = st.sidebar.selectbox(
            "Demo type:",
            [
                "Multimodal Search",
                "Text Localization",
                # "Image Captioning",
                "Image Description Generation",
                "Visual Question Answering",
                "Image Text Matching",
            ],
        )

    model_type = st.sidebar.selectbox("Model:", ["BLIP_base", "BLIP_large"])

    # raw_img = load_demo_image()

    if page == "Image Description Generation":
        show_img_caption()
    elif page == "Visual Question Answering":
        show_vqa()
    elif page == "Multimodal Search":
        show_multimodal_search()
    elif page == "Text Localization":
        show_text_localization()
    elif page == "Image Text Matching":
        show_itm()
