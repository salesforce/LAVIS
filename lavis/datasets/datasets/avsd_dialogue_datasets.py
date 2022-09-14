"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import torch
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
