"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import torch
import os
import copy
import random
from PIL import Image
from lavis.datasets.datasets.dialogue_datasets import (
    DialogueDataset,
    DialogueEvalDataset,
)


class AVSDDialDataset(DialogueDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):

        ann = self.annotation[index]

        vname = ann["image_id"]

        video = self.vis_processor(self.vis_root, vname)

        dialogue = self.text_processor(ann)

        # "image_id" is kept to stay compatible with the COCO evaluation format
        return {
            "video_fts": video["video_fts"],
            "video_token_type_ids": video["token_type_ids"],
            "input_ids": dialogue["input_ids"],
            "token_type_ids": dialogue["token_type_ids"],
            "labels": dialogue["labels"],
            "image_id": ann["image_id"],
            "instance_id": ann["instance_id"],
        }

    def collater(self, samples):

        input_ids, token_type_ids, labels, video_fts, video_token_type_ids = (
            [],
            [],
            [],
            [],
            [],
        )

        for i in samples:
            input_ids.append(i["input_ids"])
            token_type_ids.append(i["token_type_ids"])
            labels.append(i["labels"])
            video_fts.append(i["video_fts"])
            video_token_type_ids.append(i["video_token_type_ids"])

        input_ids = self.text_processor.padding(input_ids)

        labels = self.text_processor.padding(
            labels, -1
        )  # ignore token indice -1 by default
        video_fts = self.vis_processor.padding(video_fts)

        token_type_ids = self.text_processor.padding(token_type_ids)
        video_token_type_ids = self.text_processor.padding(video_token_type_ids)
        token_type_ids = torch.cat([video_token_type_ids, token_type_ids], dim=1)

        attn_mask = self.text_processor.get_attention_mask(input_ids)
        video_mask = self.vis_processor.get_attention_mask(video_fts)
        attn_mask = torch.cat([video_mask, attn_mask], dim=1)

        video_labels = (
            torch.ones((video_fts.size(0), video_fts.size(1))).long() * -1
        )  # ignore token indice -1 by default
        labels = torch.cat([video_labels, labels], dim=1)

        samples = {}
        samples["input_ids"] = input_ids
        samples["token_type_ids"] = token_type_ids
        samples["labels"] = labels
        samples["video_fts"] = video_fts
        samples["attn_mask"] = attn_mask

        return samples


class AVSDDialEvalDataset(DialogueEvalDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):

        ann = self.annotation[index]

        vname = ann["image_id"]

        video = self.vis_processor(self.vis_root, vname)

        dialogue = self.text_processor(ann)

        # "image_id" is kept to stay compatible with the COCO evaluation format
        return {
            "video_fts": video["video_fts"],
            "video_token_type_ids": video["token_type_ids"],
            "input_ids": dialogue["input_ids"],
            "token_type_ids": dialogue["token_type_ids"],
            "labels": dialogue["labels"],
            "image_id": ann["image_id"],
            "instance_id": ann["instance_id"],
        }

    def collater(self, samples):

        input_ids, token_type_ids, labels, video_fts, video_token_type_ids = (
            [],
            [],
            [],
            [],
            [],
        )

        for i in samples:
            input_ids.append(i["input_ids"])
            token_type_ids.append(i["token_type_ids"])
            labels.append(i["labels"])
            video_fts.append(i["video_fts"])
            video_token_type_ids.append(i["video_token_type_ids"])

        input_ids = self.text_processor.padding(input_ids)

        labels = self.text_processor.padding(
            labels, -1
        )  # ignore token indice -1 by default
        video_fts = self.vis_processor.padding(video_fts)

        token_type_ids = self.text_processor.padding(token_type_ids)
        video_token_type_ids = self.text_processor.padding(video_token_type_ids)
        token_type_ids = torch.cat([video_token_type_ids, token_type_ids], dim=1)

        attn_mask = self.text_processor.get_attention_mask(input_ids)
        video_mask = self.vis_processor.get_attention_mask(video_fts)
        attn_mask = torch.cat([video_mask, attn_mask], dim=1)

        video_labels = (
            torch.ones((video_fts.size(0), video_fts.size(1))).long() * -1
        )  # ignore token indice -1 by default
        labels = torch.cat([video_labels, labels], dim=1)

        samples = {}
        samples["input_ids"] = input_ids
        samples["token_type_ids"] = token_type_ids
        samples["labels"] = labels
        samples["video_fts"] = video_fts
        samples["attn_mask"] = attn_mask

        return samples


class AVSDDialInstructEvalDataset(DialogueDataset):
    def __init__(self, **kwargs):
        super().__init__(kwargs['vis_processor'], kwargs['text_processor'], kwargs['vis_root'], kwargs['ann_paths'])

        self.modalities = kwargs['modalities']

        for modality in self.modalities:
            if 'image' in modality:
                setattr(self, f"existing_{modality}_annotation",getattr(self, f'get_existing_{modality}_annotations')())
                continue
            setattr(self, f"{modality}_root", kwargs[f"{modality}_root"])
            setattr(self, f"{modality}_processor", kwargs[f"{modality}_processor"])
            setattr(self, f"existing_{modality}_annotation",getattr(self, f'get_existing_{modality}_annotations')())
        self.sample_ids = set.intersection(*[set(getattr(self, f"existing_{modality}_annotation")) for modality in self.modalities])
        self.annotation = [ann for ann in self.annotation if ann['image_id'] in self.sample_ids]
        if 'test' in kwargs['ann_paths'][0]:
             self.annotation = [ann for ann in self.annotation if ann['answer'] == '__UNDISCLOSED__']
    
    def get_existing_audio_annotations(self):
        return [f.split('.')[0] for f in os.listdir(self.audio_root)]
    
    def get_existing_video_annotations(self):
        return [f.split('.')[0] for f in os.listdir(self.video_root)]
    
    def get_audio_path(self, sample_key):
        return os.path.join(self.audio_root, sample_key) + '.mp4'
    
    def get_video_path(self, sample_key):
        return os.path.join(self.video_root, sample_key) + '.mp4'

    def __getitem__(self, index):
        ann = copy.deepcopy(self.annotation[index])
        for modality in self.modalities:
            ann[f"{modality}_path"] = getattr(self, f"get_{modality}_path")(ann['image_id'])
            
            if type(ann[f"{modality}_path"]) == list:
                ann[f"{modality}_path"] = random.choice(ann[f"{modality}_path"])
            if 'image' in modality:
                ann['image'] = self.vis_processor(Image.open(ann[f"images_path"]))
            else:
                ann[modality] = getattr(self, f"{modality}_processor")(ann[f"{modality}_path"]).to(torch.float32)
        
        ann["sample_id"] = ann["image_id"]
        dialog = ""
        for t in ann['dialog']:
            dialog += f"{t['question']} {t['answer']} "
        ann['dialog'] = dialog
        ann['text_output'] = self.text_processor(ann['answer'])
        ann['text_input'] =  self.text_processor(ann['question'])
        ann["question_id"] = index
        # ann['captions'] = ann[ann['answer']] # commented out for test dataset
        return ann
    
    def __len__(self):
        return len(self.annotation)