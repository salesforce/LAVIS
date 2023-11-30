"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
from lavis.datasets.datasets.base_dataset import BaseDataset
from lavis.datasets.datasets.caption_datasets import CaptionDataset, CaptionEvalDataset

class TextCapsCapDataset(CaptionDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        BaseDataset.__init__(self, vis_processor, text_processor, vis_root, ann_paths)
        self.annotation = self.annotation[3]['data']
        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann["image_id"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1
            ann["image"] = ann["image_id"]+'.jpg'
            ann["caption"] = ann["caption_str"]
            del ann["caption_str"]
            
class TextCapsCapInstructDataset(TextCapsCapDataset):
    def __getitem__(self, index):
        data = super().__getitem__(index)
        if data != None:
            data['text_output'] = data["text_input"]
            data['text_input'] = self.text_processor("")
        return data

class TextCapsCapEvalDataset(CaptionEvalDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        BaseDataset.__init__(self, vis_processor, text_processor, vis_root, ann_paths)
        self.annotation = self.annotation[3]['data']
        self.annotation = [ann for ann in self.annotation if "caption_str" in ann] # only keep annotations with captions

        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann["image_id"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1
            ann["image"] = ann["image_id"]+'.jpg'
            ann["caption"] = ann["caption_str"]
            del ann["caption_str"]
        self._add_instance_ids()