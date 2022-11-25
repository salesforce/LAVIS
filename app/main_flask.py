import base64
import io
import os

import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, flash, jsonify, redirect, render_template, request, url_for
from flask_cors import CORS
from PIL import Image
import matplotlib.image as pltimg
from werkzeug.utils import secure_filename
from app.text_safety_checker import handle_text

device = "cuda" if torch.cuda.is_available() else "cpu"


ALLOWED_EXTENSIONS = set(["png", "jpg", "jpeg", "gif"])


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


from lavis.models.blip_models.blip_image_text_matching import compute_gradcam
from lavis.processors import load_processor
from lavis.processors.blip_processors import (
    BlipCaptionProcessor,
    BlipImageEvalProcessor,
)
from lavis.common.utils import get_cache_path

from app.multimodal_search import (
    load_feature_extractor_model,
    load_feat,
    compute_gradcam_batch,
)
from app.utils import (
    getAttMap,
    init_bert_tokenizer,
    load_blip_itm_model,
    load_model_cache,
    read_img,
    resize_img,
)
from caption import generate_caption

app = Flask(__name__)
CORS(app)

app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024


@app.route("/api/upload", methods=["POST"])
def upload_file():
    if request.method == "POST":
        # check if the post request has the file part
        print(request.files)
        if "file" not in request.files:
            return {"status": "error", "message": "No file part"}
        file = request.files["file"]
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == "":
            return {"status": "error", "message": "No selected file", "filename": file}
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            return {
                "status": "success",
                "message": "File uploaded successfully",
                "path": os.path.join(app.config["UPLOAD_FOLDER"], filename),
            }


def decode_image0(img_obj):
    image_path = os.path.join(app.config["UPLOAD_FOLDER"], img_obj.filename)

    img_obj.save(image_path)

    img = Image.open(image_path).convert("RGB")

    # remove the image from the server
    os.remove(image_path)

    return img


def decode_image(img_obj):
    img = Image.open(img_obj).convert("RGB")
    return img


def encode_image(img_obj, encoding="ascii"):
    assert type(img_obj) == np.ndarray

    buffered = io.BytesIO()
    np.save(buffered, img_obj)
    img_str = base64.encodebytes(buffered.getvalue()).decode(encoding)

    return img_str


@app.route("/api/classification", methods=["POST"])
def classification_api():
    """
    Usage:
        curl -X POST 127.0.0.1:5000/api/classification \
            -F "image=@/path/to/image" \
            -F "class_name=merlion,elephant,giraffe,fountain,marina bay"
            -F "model_type=BLIP_base" \
            -F "score_type=Cosine"
    
    model_type: ALBEF, BLIP_base, BLIP_large, CLIP_ViT-B-32, CLIP_ViT-B-16, CLIP_ViT-L-14. Default: BLIP_base
    score_type: Cosine, Multimodal. Default: Cosine
    class_name: comma separated class names.

    """
    from app.classification import load_model_cache

    r = request

    # raw_image = r.files["image"]
    # image = decode_image(raw_image)

    request_dict = r.form.to_dict()
    image_path = request_dict.get("image", None)
    image = decode_image(image_path)
    # required fields
    model_type = request_dict.get("model_type", "BLIP_base")
    cls_names = request_dict["class_names"].split(",")
    cls_names = [handle_text(cls_name) for cls_name in cls_names]

    # optional fields
    if "CLIP" in model_type:
        score_type = "Cosine"
    else:
        score_type = request_dict.get("score_type", "Cosine")

    if model_type.startswith("BLIP"):
        text_processor = BlipCaptionProcessor(prompt="A picture of ")
        cls_prompt = [text_processor(cls_nm) for cls_nm in cls_names]

        if score_type == "Cosine":
            vis_processor = load_processor("blip_image_eval").build(image_size=224)
            img = vis_processor(image).unsqueeze(0).to(device)

            feature_extractor = load_model_cache(model_type="blip", device=device)

            sample = {"image": img, "text_input": cls_prompt}

            with torch.no_grad():
                image_features = feature_extractor.extract_features(
                    sample, mode="image"
                ).image_embeds_proj[:, 0]
                text_features = feature_extractor.extract_features(
                    sample, mode="text"
                ).text_embeds_proj[:, 0]
                sims = (image_features @ text_features.t())[0] / feature_extractor.temp

        else:
            vis_processor = load_processor("blip_image_eval").build(image_size=384)
            img = vis_processor(image).unsqueeze(0).to(device)

            model = load_blip_itm_model(device)

            output = model(img, cls_prompt, match_head="itm")
            sims = output[:, 1]

        sims = torch.nn.Softmax(dim=0)(sims)
        inv_sims = [sim * 100 for sim in sims.tolist()[::-1]]

    elif model_type.startswith("ALBEF"):
        vis_processor = load_processor("blip_image_eval").build(image_size=224)
        img = vis_processor(image).unsqueeze(0).to(device)

        text_processor = BlipCaptionProcessor(prompt="A picture of ")
        cls_prompt = [text_processor(cls_nm) for cls_nm in cls_names]

        feature_extractor = load_model_cache(model_type="albef", device=device)

        sample = {"image": img, "text_input": cls_prompt}

        with torch.no_grad():
            image_features = feature_extractor.extract_features(
                sample, mode="image"
            ).image_embeds_proj[:, 0]
            text_features = feature_extractor.extract_features(
                sample, mode="text"
            ).text_embeds_proj[:, 0]

            sims = (image_features @ text_features.t())[0] / feature_extractor.temp

        sims = torch.nn.Softmax(dim=0)(sims)
        inv_sims = [sim * 100 for sim in sims.tolist()[::-1]]

    elif model_type.startswith("CLIP"):
        if model_type == "CLIP_ViT-B-32":
            model = load_model_cache(model_type="CLIP_ViT-B-32", device=device)
        elif model_type == "CLIP_ViT-B-16":
            model = load_model_cache(model_type="CLIP_ViT-B-16", device=device)
        elif model_type == "CLIP_ViT-L-14":
            model = load_model_cache(model_type="CLIP_ViT-L-14", device=device)
        else:
            raise ValueError(f"Unknown model type {model_type}")

        if score_type == "Cosine":
            # image_preprocess = ClipImageEvalProcessor(image_size=336)
            image_preprocess = BlipImageEvalProcessor(image_size=224)
            img = image_preprocess(image).unsqueeze(0).to(device)

            sample = {"image": img, "text_input": cls_names}

            with torch.no_grad():
                clip_features = model.extract_features(sample)

                image_features = clip_features.image_embeds_proj
                text_features = clip_features.text_embeds_proj

                sims = (100.0 * image_features @ text_features.T)[0].softmax(dim=-1)
                inv_sims = sims.tolist()[::-1]

    response = {"class_names": cls_names, "scores": inv_sims}
    return response


def read_and_process_images(image_paths, vis_processor):
    raw_images = [read_img(path) for path in image_paths]
    images = [vis_processor(r_img) for r_img in raw_images]
    images_tensors = torch.stack(images).to(device)

    return raw_images, images_tensors


def convert_to_bool(value):
    if isinstance(value, bool):
        return value

    if value.lower() == "true":
        return True
    elif value.lower() == "false":
        return False
    else:
        raise ValueError(f"Unknown value {value}")


@app.route("/api/search", methods=["POST"])
def search_api():
    """
    Usage:
        curl -X POST 127.0.0.1:5000/api/search \
            -F model_type=BLIP_base \
            -F query="A dog running in the grass" \
            -F num_display=12 \
            -F itm_ranking=False \
            -F show_gradcam=False

    """
    file_root = get_cache_path("coco/images/train2014")

    r = request

    request_dict = r.form.to_dict()
    query = handle_text(request_dict["query"])
    num_display = int(request_dict.get("num_display", 12))
    show_gradcam = convert_to_bool(request_dict.get("show_gradcam", False))
    itm_ranking = convert_to_bool(request_dict.get("itm_ranking", False))
    model_type = request_dict.get("model_type", "BLIP_base")

    vis_processor = load_processor("blip_image_eval").build(image_size=384)
    text_processor = load_processor("blip_caption")

    user_question = text_processor(query)
    feature_extractor = load_feature_extractor_model(device)

    # ======= ITC =========
    sample = {"text_input": user_question}

    with torch.no_grad():
        text_feature = feature_extractor.extract_features(
            sample, mode="text"
        ).text_embeds_proj[0, 0]

        path2feat, paths, all_img_feats = load_feat()
        all_img_feats.to(device)
        all_img_feats = F.normalize(all_img_feats, dim=1)

        similarities = text_feature @ all_img_feats.T
        indices = torch.argsort(similarities, descending=True)[:num_display]

    top_paths = [paths[ind.detach().cpu().item()] for ind in indices]
    sorted_similarities = [similarities[idx] for idx in indices]
    filenames = [os.path.join(file_root, p) for p in top_paths]

    # ========= ITM and GradCam ==========
    bsz = 4  # max number of images to avoid cuda oom
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

    if show_gradcam:
        images_to_show = avg_gradcams
    else:
        images_to_show = [np.float32(r_img) / 255 for r_img in all_raw_images]

    # save images to disk for local viewing
    for i, img in enumerate(images_to_show):
        img = np.clip(img, 0, 1)
        with open(f"static/{i}.jpg", "wb") as f:
            pltimg.imsave(f, img)

    images = [encode_image(img) for img in images_to_show]

    return {"images": images}


@app.route("/api/vqa", methods=["POST"])
def vqa_api():
    """VQA API

    Usage:
        curl -X POST 127.0.0.1:5000/api/vqa -F "image=@/path/to/image" -F "question=What is this?"

    Only support BLIP vqa model for now.
    """
    r = request

    # raw_image = r.files["image"]

    # image = decode_image(raw_image)

    request_dict = r.form.to_dict()
    image_path = request_dict.get("image", None)
    image = decode_image(image_path)
    question = request_dict["question"]
    question = handle_text(question)

    model = load_model_cache(
        "blip_vqa", model_type="vqav2", is_eval=True, device=device
    )

    vis_processor = load_processor("blip_image_eval").build(image_size=480)
    txt_processor = load_processor("blip_question").build()

    image = vis_processor(image).unsqueeze(0).to(device)
    question = txt_processor(question)

    samples = {"image": image, "text_input": [question]}

    answers = model.predict_answers(samples, inference_method="generate")

    return {"answer": handle_text(answers[0])}


@app.route("/api/text_localization", methods=["POST"])
def text_localization_api():
    """
    Usage:
        curl -X POST 127.0.0.1:5000/api/text_localization \
            -F "image=@/path/to/image" \
            -F "query=A man is riding a bike." \
            -F "model_type=BLIP_large" \
            -F "layer_number=7"

    model_type: BLIP_base, BLIP_large, default BLIP_large.
    layer_number: integer in [0, 10], default 7

    Return:
        A dictionary with keys:
            - token_gradcam: token level gradcam, list of b64 encoded image.
                            This should have the same length as the words.
            - words: list of words to visualize. This should have the same length as the token_gradcam.
            - avg_gradcam: average gradcam of the words, b64 encoded image.
    """
    r = request

    # raw_image = r.files["image"]

    request_dict = r.form.to_dict()

    # required fields
    query = request_dict["query"]
    query = handle_text(query)

    request_dict = r.form.to_dict()
    image_path = request_dict.get("image", None)
    image = decode_image(image_path)
    vis_processor = load_processor("blip_image_eval").build(image_size=384)
    text_processor = load_processor("blip_caption")

    # optional fields
    model_type = request_dict.get("model_type", "BLIP_large")
    layer_num = int(request_dict.get("layer_num", 7))

    if model_type.startswith("BLIP"):
        blip_type = model_type.split("_")[1]
        model = load_blip_itm_model(device, model_type=blip_type)

    img = vis_processor(image).unsqueeze(0).to(device)
    qry = text_processor(query)

    # words
    qry_tok = bert_tokenizer(qry, return_tensors="pt").to(device)
    words = bert_tokenizer.decode(qry_tok.input_ids[0][1:-1]).split()
    # print(words)

    # compute gradcam
    w, h = image.size
    scaling_factor = 720 / w
    resized_image = image.resize((int(w * scaling_factor), int(h * scaling_factor)))
    norm_img = np.float32(resized_image) / 255

    gradcam, _ = compute_gradcam(model, img, qry, qry_tok, block_num=layer_num)
    gradcam = [getAttMap(norm_img, gc, blur=True) for gc in gradcam[0][1:-1]]

    for i, img in enumerate(gradcam):
        img = np.clip(img, 0, 1)
        with open(f"static/tl_{i}.jpg", "wb") as f:
            pltimg.imsave(f, img)

    gradcam_img = [encode_image(img) for img in gradcam]

    # avg_grad cam should be displayed standalone
    # token_gradcam should be displayed accompanied with words
    avg_gradcam = gradcam_img[0]
    tok_gradcam = gradcam_img[2:-1]

    return jsonify(
        {"token_gradcam": tok_gradcam, "avg_gradcam": avg_gradcam, "word": words}
    )


@app.route("/api/caption", methods=["POST"])
def generate_caption_api():
    """
    Usage:
    curl -X POST 127.0.0.1:5000/api/caption \
                -F "image=@/path/to/image.jpg" \
                -F "model_type=large" \
                -F "use_nucleus_sampling=False" \
                -F "max_length=40" \
                -F "min_length=5" \
                -F "num_captions=5"
    
    model_type: BLIP_base, BLIP_large
    use_nucleus_sampling: True, False. If True, use nucleus sampling. Otherwise, use beam search.
    max_length: int
    min_length: int
    num_captions: int
    
    Returns: a dictionary with the key "caption" and the value is a list of captions
    """

    r = request

    # raw_image = r.files["image"]

    # required fields
    # decode image from form data

    request_dict = r.form.to_dict()

    # optional fields
    model_type = request_dict.get("model_type", "large")
    use_nucleus_sampling = convert_to_bool(
        request_dict.get("use_nucleus_sampling", False)
    )
    num_beams = int(request_dict.get("num_beams", 3))
    max_length = int(request_dict.get("max_length", 40))
    min_length = int(request_dict.get("min_length", 5))
    num_captions = int(request_dict.get("num_captions", 1))
    image_path = request_dict.get("image", None)
    image = decode_image(image_path)
    # load model
    app.logger.info("Loading model...")
    model = load_model_cache(
        "blip_caption",
        model_type=f"{model_type}_coco",
        is_eval=True,
        device=device,
    )
    app.logger.info("Model loaded.")

    # # load processors
    vis_processor = load_processor("blip_image_eval").build(image_size=384)

    # load images
    image = vis_processor(image).unsqueeze(0).to(device)

    # generate caption
    captions = generate_caption(
        model=model,
        image=image,
        use_nucleus_sampling=use_nucleus_sampling,
        num_beams=num_beams,
        max_length=max_length,
        min_length=min_length,
        num_captions=num_captions,
    )
    captions = [handle_text(c) for c in captions]

    return {"caption": captions}


if __name__ == "__main__":
    app.debug = True

    # UPLOAD FOLDER
    app.config["UPLOAD_FOLDER"] = "static/uploads"

    # make upload folder if not exists
    if not os.path.exists(app.config["UPLOAD_FOLDER"]):
        os.makedirs(app.config["UPLOAD_FOLDER"])

    bert_tokenizer = init_bert_tokenizer()

    app.run(debug=True)
