"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import PIL
from PIL import Image
import numpy as np

from lavis.common.annotator.util import resize_image, HWC3
from lavis.common.annotator.canny import CannyDetector

apply_canny = CannyDetector()


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def preprocess_canny(
    input_image: np.ndarray,
    image_resolution: int,
    low_threshold: int,
    high_threshold: int,
):
    image = resize_image(HWC3(input_image), image_resolution)
    control_image = apply_canny(image, low_threshold, high_threshold)
    control_image = HWC3(control_image)
    # vis_control_image = 255 - control_image
    # return PIL.Image.fromarray(control_image), PIL.Image.fromarray(
    #     vis_control_image)
    return PIL.Image.fromarray(control_image)


def generate_canny(cond_image_input, low_threshold, high_threshold):
    # convert cond_image_input to numpy array
    cond_image_input = np.array(cond_image_input).astype(np.uint8)

    # canny_input, vis_control_image = preprocess_canny(cond_image_input, 512, low_threshold=100, high_threshold=200)
    vis_control_image = preprocess_canny(cond_image_input, 512, low_threshold=low_threshold, high_threshold=high_threshold)

    return vis_control_image 