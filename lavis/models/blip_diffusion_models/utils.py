"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import numpy as np
import PIL
import torch
from diffusers.utils.pil_utils import PIL_INTERPOLATION
from PIL import Image

from lavis.common.annotator.canny import CannyDetector
from lavis.common.annotator.util import HWC3, resize_image

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


def prepare_cond_image(
        image, width, height, batch_size, device, do_classifier_free_guidance=True
    ):
        if not isinstance(image, torch.Tensor):
            if isinstance(image, Image.Image):
                image = [image]

            if isinstance(image[0], Image.Image):
                images = []

                for image_ in image:
                    image_ = image_.convert("RGB")
                    image_ = image_.resize(
                        (width, height), resample=PIL_INTERPOLATION["lanczos"]
                    )
                    image_ = np.array(image_)
                    image_ = image_[None, :]
                    images.append(image_)

                image = images

                image = np.concatenate(image, axis=0)
                image = np.array(image).astype(np.float32) / 255.0
                image = image.transpose(0, 3, 1, 2)
                image = torch.from_numpy(image)
            elif isinstance(image[0], torch.Tensor):
                image = torch.cat(image, dim=0)

        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            # repeat_by = num_images_per_prompt
            raise NotImplementedError

        image = image.repeat_interleave(repeat_by, dim=0)

        # image = image.to(device=self.device, dtype=dtype)
        image = image.to(device=device)

        if do_classifier_free_guidance:
            image = torch.cat([image] * 2)

        return image
