"""
 # Copyright (c) 2022, salesforce.com, inc.
 # All rights reserved.
 # SPDX-License-Identifier: BSD-3-Clause
 # For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from PIL import Image
import requests
import torch

import os

from lavis.common.registry import registry
from lavis.processors import *
from lavis.models import *
from lavis.common.utils import build_default_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_demo_image():
    img_url = (
        "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"
    )
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")

    return raw_image


def read_img(filepath):
    raw_image = Image.open(filepath).convert("RGB")

    return raw_image


# model
model_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base.pth"
feature_extractor = BlipFeatureExtractor(pretrained=model_url)

feature_extractor.eval()
feature_extractor = feature_extractor.to(device)

# preprocessors
vis_processor = BlipImageEvalProcessor(image_size=224)
text_processor = BlipCaptionProcessor()

# files to process
# file_root = "/export/home/.cache/lavis/coco/images/val2014"
file_root = "/export/home/.cache/lavis/coco/images/train2014"
filepaths = os.listdir(file_root)

print(len(filepaths))

caption = "dummy"

path2feat = dict()
bsz = 256

images_in_batch = []
filepaths_in_batch = []

for i, filename in enumerate(filepaths):
    if i % bsz == 0 and i > 0:
        images_in_batch = torch.cat(images_in_batch, dim=0).to(device)
        with torch.no_grad():
            image_features = feature_extractor(
                images_in_batch, caption, mode="image", normalized=True
            )[:, 0]

        for filepath, image_feat in zip(filepaths_in_batch, image_features):
            path2feat[os.path.basename(filepath)] = image_feat.detach().cpu()

        images_in_batch = []
        filepaths_in_batch = []

        print(len(path2feat), image_features.shape)
    else:
        filepath = os.path.join(file_root, filename)

        image = read_img(filepath)
        image = vis_processor(image).unsqueeze(0)

        images_in_batch.append(image)
        filepaths_in_batch.append(filepath)

torch.save(path2feat, "path2feat_coco_train2014.pth")
