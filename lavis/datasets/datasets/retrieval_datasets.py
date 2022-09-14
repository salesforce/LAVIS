"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
from collections import OrderedDict

from lavis.datasets.datasets.base_dataset import BaseDataset
from PIL import Image


class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]
        visual_key = "image" if "image" in ann else "video"

        return OrderedDict(
            {
                "file": ann[visual_key],
                "caption": ann["caption"],
                visual_key: sample[visual_key],
            }
        )


class RetrievalDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann["image_id"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

    def __getitem__(self, index):

        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        caption = self.text_processor(ann["caption"])

        return {
            "image": image,
            "text_input": caption,
            "image_id": self.img_ids[ann["image_id"]],
            "instance_id": ann["instance_id"],
        }


class RetrievalEvalDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """

        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}

        txt_id = 0
        for img_id, ann in enumerate(self.annotation):
            self.image.append(ann["image"])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann["caption"]):
                self.text.append(self.text_processor(caption))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1

    def __getitem__(self, index):

        image_path = os.path.join(self.vis_root, self.annotation[index]["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)

        return {"image": image, "index": index}


class VideoRetrievalDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of videos.
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann["video"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

    def __getitem__(self, index):

        ann = self.annotation[index]

        vpath = os.path.join(self.vis_root, ann["video"])

        video = self.vis_processor(vpath)
        caption = self.text_processor(ann["caption"])

        # return image, caption, self.img_ids[ann['image_id']]
        return {
            "video": video,
            "text_input": caption,
            "image_id": self.img_ids[ann["video"]],
        }


class VideoRetrievalEvalDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of videos.
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """

        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}

        txt_id = 0
        for img_id, ann in enumerate(self.annotation):
            self.image.append(ann["video"])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann["caption"]):
                self.text.append(self.text_processor(caption))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1

    def __getitem__(self, index):
        ann = self.annotation[index]

        vpath = os.path.join(self.vis_root, ann["video"])
        video = self.vis_processor(vpath)

        return {"video": video, "index": index}
