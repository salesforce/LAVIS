"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import copy
import os
from lavis.datasets.datasets.audio_captioning_datasets import AudioCapsDataset
from lavis.datasets.datasets.base_dataset import BaseDataset
import torch
import random
from collections import Counter

class AudioCapsQADataset(AudioCapsDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_binary = kwargs.get('add_binary', False)
        self.binary_templates = ["do you hear {}?", "is this {}?", "does the audio contain {}?"]
    
    def __getitem__(self, index):
        ann = copy.deepcopy(self.annotation[index])
        for modality in self.modalities:
            if modality == 'audio' and self.cached:
                ann[f"{modality}_path"] = getattr(self, f"get_cached_{modality}_path")(ann)
                ann["audio"] = torch.load(ann[f"{modality}_path"])
            else:
                ann[f"{modality}_path"] = getattr(self, f"get_{modality}_path")(ann)
                if isinstance(ann[f"{modality}_path"], list):
                    ann[f"{modality}_path"] = random.choice(ann[f"{modality}_path"])
                ann[modality if 'image' not in modality else 'image'] = getattr(self, f"{'vis' if 'image' in modality else modality}_processor")(ann[f"{modality}_path"])
        
        if ann["audio"].sum() == 0:
            return None
        if self.add_binary and random.randint(0,10) < 3:
            yes_answer = random.randint(0,10)<5
            if not yes_answer:
                caption_index = random.choice(list(set(range(len(self.annotation))).difference(set([index]))))
                caption = self.annotation[caption_index]['caption']
            else:
                caption = ann['caption']
            
            question = random.choice(self.binary_templates).format(caption)
            answer = 'yes' if yes_answer else 'no'
            return {
                "text_input": self.text_processor(question),
                "instance_id": ann["instance_id"],
                "text_output":answer,
                "answer":answer,
                "caption": ann['caption'],
                "audio": ann['audio'],
                "audio_id": ann['youtube_id'],
                "question_id": ann['youtube_id'],
            }

        return {
            "text_input": self.text_processor(ann['question']),
            "instance_id": ann["instance_id"],
            "text_output":ann['answer'],
            "answer":ann['answer'],
            "caption": ann['caption'],
            "audio": ann['audio'],
            "audio_id": ann['youtube_id'],
            "question_id": ann['youtube_id'],
        }



class ClothoQADataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(kwargs['vis_processor'], kwargs['text_processor'], kwargs['vis_root'], kwargs['ann_paths'])

        self.non_binary_cls = kwargs.get('non_bin',False)
        self.unanimous = kwargs.get('unanimous',False)
        

        annotation = []
        for i in range(0, len(self.annotation), 3):
            new_ann = self.annotation[i]
            new_ann['question'] = new_ann['QuestionText']
            del new_ann['QuestionText']
            new_ann['answer'] = [self.annotation[i+off]['answer'] for off in range(3)]
            if self.unanimous and Counter(new_ann['answer'])[new_ann['answer'][0]] != 3:
                continue
            if self.non_binary_cls and ('yes' in new_ann['answer'] or 'no' in new_ann['answer']):
                continue
            new_ann["question_id"] = new_ann['instance_id']
            annotation.append(new_ann)
        self.modalities = kwargs['modalities']
        for modality in self.modalities:
            setattr(self, f"{modality}_root", kwargs[f"{modality}_root"])
            setattr(self, f"{modality}_processor", kwargs[f"{modality}_processor"])
        self.annotation = annotation

    
    def __getitem__(self, index):
        ann = copy.deepcopy(self.annotation[index])
        audio_path = os.path.join(self.audio_root, ann["file_name"])
        ann['audio'] = self.audio_processor(audio_path)
       
        if ann["audio"].sum() == 0:
            return None

        return {
            "text_input": self.text_processor(ann['question']),
            "question": self.text_processor(ann['question']),
            "instance_id": ann["instance_id"],
            "text_output":random.choice(ann['answer']),
            "answer":ann['answer'],
            "answers":ann['answer'],
            "audio": ann['audio'],
            "question_id": ann['instance_id'],
        }
    
    def _build_templates(self, template):
        return None
