"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
from collections import OrderedDict
from PIL import Image
import copy

from lavis.datasets.datasets.base_dataset import BaseDataset
from lavis.common.utils import is_serializable


class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        serializable_sample_keys = [k for k,v in sample.items() if is_serializable(v)]
        serializable_ann_keys = [k for k,v in ann.items() if is_serializable(v)]
        display = {k:sample[k] for k in serializable_sample_keys}
        display.update({k:ann[k] for k in serializable_ann_keys})

        return OrderedDict(
            display
        )



class DisCRnDataset(BaseDataset, __DisplMixin):
    def __init__(self, **kwargs):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        pc_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file 
        """
        super().__init__(kwargs['vis_processor'], kwargs['text_processor'], kwargs['vis_root'], kwargs['ann_paths'])

        self.ds_name = kwargs['dataset_name']
        self.modalities = [str(m) for m in kwargs['modalities']]
        ## from lavis convention, sometimes "image" modality is denoted as images
        if "images" in self.modalities:
            self.modalities[self.modalities.index("images")] = "image"
        self.npoints = 8192
        self.sample_points_num = self.npoints
        self.annotation = self.annotation
        self.view = kwargs.get('view', 2)
        self.classnames = copy.deepcopy(self.modalities)
        self.classnames = kwargs.get('classnames', ["first", "second"])
        self.total = kwargs.get('total', 'all')
        self.ground_truth = kwargs.get('ground_truth', False)
        self.shuffle_modalities = kwargs.get('shuffle_modalities', False)
        self.balance_labels = kwargs.get('balance_labels', True)
        self.raw = kwargs.get('raw', False)

        if self.total != 'all':
            self.annotation = self.annotation[:self.total]
        
        for modality in self.modalities:
            if "image" not in modality:
                setattr(self, f"{modality}_root", kwargs[f"{modality}_root"])
                setattr(self, f"{modality}_processor", kwargs[f"{modality}_processor"])
            setattr(self, f"existing_{modality}_annotation",getattr(self, f'get_existing_{modality}_annotations')())
        
        self.sample_ids = set.intersection(*[set(getattr(self, f"existing_{modality}_annotation")) for modality in self.modalities])
        self.annotation = [ann for ann in self.annotation if ann['sample_ids'][0] in self.sample_ids and ann['sample_ids'][1] in self.sample_ids]
        self._add_instance_ids()

    def get_existing_image_annotations(self):
        if self.ds_name == 'objaverse':
            return [f.split('_')[0] for f in os.listdir(os.path.join(self.vis_root, f'compressed_imgs_view{self.view}/Cap3D_imgs_view{self.view}/'))]
    
    def get_image_path(self, ann, entity_index):
        if self.ds_name == 'objaverse':
            # data downloaded from: https://huggingface.co/datasets/tiange/Cap3D/tree/main/RenderedImage_zips
            return os.path.join(self.vis_root, f'compressed_imgs_view{self.view}/Cap3D_imgs_view{self.view}/', ann['sample_ids'][entity_index]+f'_{self.view}.jpeg')

    def get_existing_audio_annotations(self):
        return [f.split('_')[0] for f in os.listdir(self.audio_root)]
    
    def get_audio_path(self, ann, entity_index):
        if self.ds_name == 'audiocaps':
            return str(os.path.join(self.audio_root, ann['sample_ids'][entity_index] + '_{}.flac'.format(int(ann['start_seconds'][entity_index]))))
    
    def get_video_path(self, ann, entity_index):
        if self.ds_name == 'audiocaps':
            return str(os.path.realpath(os.path.join(self.video_root,ann['sample_ids'][entity_index] + '_{}.mp4'.format(int(ann['start_seconds'][entity_index])))))

    def get_existing_video_annotations(self):
        return [f.split('_')[0] for f in os.listdir(self.video_root)]
    
    def get_existing_pc_annotations(self):
        if self.ds_name == 'objaverse':
            return os.listdir(self.pc_root)

    def get_pc_path(self, ann, entity_index):
        if self.ds_name == 'objaverse':
            return os.path.join(self.pc_root, ann['sample_ids'][entity_index], '{}_{}.npz'.format(ann['sample_ids'][entity_index], self.npoints))

    def __getitem__(self, index):
        ann = copy.deepcopy(self.annotation[index])
        N = 2 # number of inputs
        ann["question_id"] = ann["instance_id"]
        ann[f"modalities"] = copy.deepcopy(self.modalities)
        for i,modality in enumerate(self.modalities):
            if  ann[f'captions_pred_{modality}'] == None or ann[f'captions_pred_{modality}'][i]== None:
                        return None
        if len(self.modalities) == 1: # both modalities of the same type.
            ann[f"modalities"] = [self.modalities[0]] * N
        
        if self.balance_labels:
            if (index%2 and ann["label"] == 1) or (not index%2 and ann['label'] == 0):
                ann["label"] = 1- ann["label"] 
                ann["properties"] = [ann['properties'][1],ann['properties'][0]]
                ann["captions"] = [ann['captions'][1],ann['captions'][0]]
                if self.shuffle_modalities:
                    ann['modalities'] = [ann['modalities'][1],ann['modalities'][0]] # if we comment this out, we can have batch size > 1. Maintaining for reproducibility.
                for modality in self.modalities:
                    ann[f'captions_pred_{modality}'] = [ann[f'captions_pred_{modality}'][1], ann[f'captions_pred_{modality}'][0]]
        
        ## baseline captions
        ann["baseline_captions"] = [c for c in ann["captions"]] if self.ground_truth else [ann[f'captions_pred_{ann["modalities"][0]}'][0], ann[f'captions_pred_{ann["modalities"][1]}'][1]]
        # ann["baseline_captions"] = [c.replace('..', '.') for c in ann["baseline_captions"]]
        ann["baseline_captions"] = [c.strip() if c!=None else "" for c in ann["baseline_captions"]]
        ## text input
        ann["text_input"] = self.text_processor(f'{ann["question"].replace("which entity", "which of the two options").replace("which object", "which of the two options").replace("which image", "which of the two options").replace("which audio", "which of the two options").replace("audio", "object").replace("image", "object")}?'.replace('??', '?'))
        # ann["text_input"] = self.text_processor(f'{ann["question"]}?'.replace('??', '?'))
        ## answers
        first_answers = [ann['modalities'][0], "the first option.", "the first", "left one", "(a) left",  "(a) left one", "(a)", 'a.', 'A.', "a)", "(A)", 'Input A', 'Entity 1', 'Object 1','Entity A', 'Object A', 'left', 'first', '1st', 'input 1', '1','a', 'input a', "the first", "the left one"]
        second_answers = [ann['modalities'][1], "the second option.", "the second.", "second option", "the second option", "second option.", "right one","(b) right", "(b) right one" , "(b)", "b)", 'Input B', 'right', 'second', '2nd', 'input 2', '2', 'b', 'input b', 'Object 2','Entity B', 'Object B', "the second", "the right one", "the second one"]
        if ann["label"] == 0:
            ann["answers"] = first_answers
        else:
            ann["answers"] = second_answers 
        if 'pc' in ann["answers"]:
            ann["answers"].extend(['3d', '3d model', 'model', 'rendering', 'a 3d', 'a 3d model'])
        if 'image' in ann["answers"]:
            ann["answers"].extend(['photo', 'picture'])
        if 'audio' in ann["answers"]:
            ann["answers"].append('sound')
        ## label
        ann["label"] = self.classnames[ann["label"]]
        ann['answer'] = ann["answers"] # for vqa task compatibility

        ## get data
        for i,modality in enumerate(ann["modalities"]):
            path = getattr(self, f"get_{modality}_path")(ann, i)
            if 'image' in modality:
                path = Image.open(path).convert("RGB")
            if self.raw:
                ann[modality] = path
                continue
            try:
                ann[modality] = getattr(self, f"{'vis' if 'image' in modality else modality}_processor")(path)
            except:
                return None
        
        ann["discrn"] = True # signify to model, this is a discrn task
     
        return ann
    
    def __len__(self):
        return len(self.annotation)