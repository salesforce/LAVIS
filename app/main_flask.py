import base64
import io
import os
import torch
from flask import Flask, request, jsonify, flash, redirect, url_for, render_template
from flask_cors import CORS
import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename

device = "cuda" if torch.cuda.is_available() else "cpu"


ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

from app.utils import (
    getAttMap,
    init_bert_tokenizer,
    load_blip_itm_model,
    load_model_cache,
)
from caption import generate_caption
from lavis.models.blip_models.blip_image_text_matching import compute_gradcam
from lavis.processors import load_processor
from lavis.processors.blip_processors import (
    BlipCaptionProcessor,
    BlipImageEvalProcessor,
)

app = Flask(__name__)
CORS(app)

app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024


@app.route("/api/upload", methods=["POST"])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        print(request.files)
        if 'file' not in request.files:
            return {'status': "error", 'message': "No file part"}
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            return  {'status': "error", 'message': "No selected file", 'filename': file}
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return {'status': "success", 'message': "File uploaded successfully"}



def decode_image(img_obj):
    image_path = os.path.join(app.config["UPLOAD_FOLDER"], img_obj.filename)

    img_obj.save(image_path)

    img = Image.open(image_path).convert("RGB")

    # remove the image from the server
    os.remove(image_path)

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

    raw_image = r.files["image"]
    image = decode_image(raw_image)

    request_dict = r.form.to_dict()

    # required fields
    model_type = request_dict.get("model_type", "BLIP_base")
    cls_names = request_dict["class_names"].split(",")

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


@app.route("/api/vqa", methods=["POST"])
def vqa_api():
    """VQA API

    Usage:
        curl -X POST 127.0.0.1:5000/api/vqa -F "image=@/path/to/image" -F "question=What is this?"

    Only support BLIP vqa model for now.
    """
    r = request

    raw_image = r.files["image"]

    image = decode_image(raw_image)

    request_dict = r.form.to_dict()

    question = request_dict["question"]

    model = load_model_cache(
        "blip_vqa", model_type="vqav2", is_eval=True, device=device
    )

    vis_processor = load_processor("blip_image_eval").build(image_size=480)
    txt_processor = load_processor("blip_question").build()

    image = vis_processor(image).unsqueeze(0).to(device)
    question = txt_processor(question)

    samples = {"image": image, "text_input": [question]}

    answers = model.predict_answers(samples, inference_method="generate")

    return {"answer": answers[0]}


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

    raw_image = r.files["image"]

    request_dict = r.form.to_dict()

    # required fields
    query = request_dict["query"]

    # decode image from form data
    image = decode_image(raw_image)
    request_dict = r.form.to_dict()

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
    words = [bert_tokenizer.decode(qk) for qk in qry_tok.input_ids[0][1:-1]]
    # print(words)

    # compute gradcam
    w, h = image.size
    scaling_factor = 720 / w
    resized_image = image.resize((int(w * scaling_factor), int(h * scaling_factor)))
    norm_img = np.float32(resized_image) / 255

    gradcam, _ = compute_gradcam(model, img, qry, qry_tok, block_num=layer_num)
    gradcam_img = [
        encode_image(getAttMap(norm_img, gc, blur=True)) for gc in gradcam[0][1:-1]
    ]

    # avg_grad cam should be displayed standalone
    # token_gradcam should be displayed accompanied with words
    avg_gradcam = gradcam_img[0]
    tok_gradcam = gradcam_img[1:]

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

    raw_image = r.files["image"]

    # required fields
    # decode image from form data
    image = decode_image(raw_image)

    request_dict = r.form.to_dict()

    # optional fields
    model_type = request_dict.get("model_type", "large")
    use_nucleus_sampling = bool(request_dict.get("use_nucleus_sampling", False))
    num_beams = int(request_dict.get("num_beams", 3))
    max_length = int(request_dict.get("max_length", 40))
    min_length = int(request_dict.get("min_length", 5))
    num_captions = int(request_dict.get("num_captions", 1))

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

    return {"caption": captions}





if __name__ == "__main__":
    app.debug = True

    # UPLOAD FOLDER
    app.config["UPLOAD_FOLDER"] = "uploads"

    # make upload folder if not exists
    if not os.path.exists(app.config["UPLOAD_FOLDER"]):
        os.makedirs(app.config["UPLOAD_FOLDER"])

    bert_tokenizer = init_bert_tokenizer()

    app.run(debug=True)
