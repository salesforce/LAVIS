import os, shutil

import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from app import cache_root, device, job_output_path, finished_job_path
from app.utils import (
    getAttMap,
    init_bert_tokenizer,
    load_blip_itm_model,
    read_img,
    resize_img,
    get_pending_jobs,
    create_uniq_user_job_name
)
from lavis.models import BlipFeatureExtractor, load_model
from lavis.processors import load_processor

if torch.cuda.is_available():
    torch.cuda.set_device(0)
    device = "cuda"
else:
    device = "cpu"

job_type = 'search'

def load_feat():
    from lavis.common.utils import download_url

    dirname = os.path.join(os.path.dirname(__file__), "assets")
    filename = "path2feat_coco_train2014.pth"
    filepath = os.path.join(dirname, filename)
    url = "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/assets/path2feat_coco_train2014.pth"

    if not os.path.exists(filepath):
        download_url(url=url, root=dirname, filename="path2feat_coco_train2014.pth")

    path2feat = torch.load(filepath)
    paths = sorted(path2feat.keys())

    all_img_feats = torch.stack([path2feat[k] for k in paths], dim=0).to(device)

    return path2feat, paths, all_img_feats

def load_feature_extractor_model(device):
    model_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base.pth"

    model = load_model("blip_feature_extractor", model_type="base", is_eval=True, device=device)
    model.load_from_pretrained(model_url)

    return model

def search(time_stamp, user_question, feature_extractor, vis_processor, raw_user_question, num_display, itm_model):
    sample = {"text_input": user_question}
    with torch.no_grad():
        text_feature = feature_extractor.extract_features(
            sample, mode="text").text_embeds_proj[0, 0]

        path2feat, paths, all_img_feats = load_feat()
        all_img_feats.to(device)
        all_img_feats = F.normalize(all_img_feats, dim=1)

        num_cols = 4
        num_rows = int(num_display) // num_cols

        similarities = text_feature @ all_img_feats.T
        indices = torch.argsort(similarities, descending=True)[:num_display]

    top_paths = [paths[ind.detach().cpu().item()] for ind in indices]
    sorted_similarities = [similarities[idx] for idx in indices]
    file_root = os.path.join(cache_root, "coco/images/train2014/")
    filenames = [os.path.join(file_root, p) for p in top_paths]
    outpath = os.path.join(job_output_path, job_type)
    os.makedirs(outpath, exist_ok=True)

    bsz = 8  # max number of images to avoid cuda oom

    #itm_model = load_blip_itm_model("cuda", model_type=blip_type)

    tokenizer = init_bert_tokenizer()
    queries_batch = [user_question] * bsz
    queries_tok_batch = tokenizer(queries_batch, return_tensors="pt").to("cpu")

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

    #avg_gradcams = torch.cat(avg_gradcams)
    #all_raw_images = torch.cat(all_raw_images)

    itm_scores = torch.cat(itm_scores)[:, 1]
    torch.save(itm_scores, outpath+'/{}_itm.pt'.format(create_uniq_user_job_name(time_stamp,raw_user_question)))
    np.save(outpath+'/{}_avg_gradcams.npy'.format(create_uniq_user_job_name(time_stamp,raw_user_question)), avg_gradcams, allow_pickle=True)
    np.save(outpath+'/{}_all_raw_images.npy'.format(create_uniq_user_job_name(time_stamp,raw_user_question)),all_raw_images,allow_pickle=True)

    search_result = outpath+'/{}_result.txt'.format(create_uniq_user_job_name(time_stamp,raw_user_question))
    with open(search_result,'w') as f:
        for filename in filenames:
            f.write(filename+'\n')

def back_end():
    # === event ===
    vis_processor = load_processor("blip_image_eval").build(image_size=384)
    text_processor = load_processor("blip_caption")
    feature_extractor = load_feature_extractor_model(device)
    os.makedirs("{}/{}/".format(finished_job_path, job_type), exist_ok=True)
    large_itm_model = load_blip_itm_model(device, model_type='large')
    base_itm_model = load_blip_itm_model(device, model_type='base')

    while True:
        pending_jobs = get_pending_jobs(job_type)
        for job in pending_jobs:
            while True:
                with open(job) as f:
                    content = f.readline().rstrip(' \n')
                    if len(content.split('\t')) == 4: break
            time_stamp, raw_user_question, num_display, blip_type = content.split('\t')
            user_question = text_processor(raw_user_question)
            if blip_type == 'large':
                search(time_stamp, user_question, feature_extractor, vis_processor, raw_user_question, int(num_display), large_itm_model)
            else:
                search(time_stamp, user_question, feature_extractor, vis_processor, raw_user_question, int(num_display), base_itm_model)
            shutil.move(job, "{}/{}/".format(finished_job_path,job_type))


def read_and_process_images(image_paths, vis_processor):
    raw_images = [read_img(path) for path in image_paths]
    images = [vis_processor(r_img) for r_img in raw_images]
    images_tensors = torch.stack(images).to(device)

    return raw_images, images_tensors


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
        ).to(device=device) # (bsz,1,token_len, 1,1)
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

if __name__ == '__main__':
    back_end()
