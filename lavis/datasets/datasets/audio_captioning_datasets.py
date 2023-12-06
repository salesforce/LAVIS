"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
from collections import OrderedDict
import torch
import copy
import pathlib
import random
import json
import pandas as pd
import torchaudio
import torch
from tqdm import tqdm

from lavis.datasets.datasets.base_dataset import BaseDataset

class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": ann["image"],
                "label": ann["caption"],
                "audio": sample["audio"],
                "audio_path": sample["audio_path"],
                "caption": sample["caption"],
    
            }
        )


class AudioCaptioningDataset(BaseDataset, __DisplMixin):
    def __init__(self, **kwargs):
        self.modalities = kwargs['modalities']
        super().__init__(kwargs['vis_processor'], kwargs['text_processor'], kwargs['vis_root'], kwargs['ann_paths'])
        for modality in self.modalities:
            setattr(self, f"{modality}_root", kwargs[f"{modality}_root"])
            setattr(self, f"{modality}_processor", kwargs[f"{modality}_processor"])
            setattr(self, f"existing_{modality}_annotation",getattr(self, f'get_existing_{modality}_annotations')())
        
    def get_audio_path(self, ann):
        raise NotImplementedError("Subclasses should implement this!")
    
    def is_empty_audio(self, ann):
        path = self.get_audio_path(ann)
        try:
            waveform, sr = torchaudio.load(path)

            # Convert to mono if it's stereo
            if waveform.shape[0] == 2:
                waveform = torch.mean(waveform, dim=0)

        except torchaudio.TorchaudioException:
            return True  # Audio loading failed

        return waveform.nelement() == 0
    
    def get_existing_audio_annotations(self):
        return [f.split('_')[0] for f in os.listdir(self.audio_root)]

    def get_existing_video_annotations(self):
        return os.listdir(self.video_root)
    
    def get_existing_images_annotations(self):
        return os.listdir(self.vis_root)
    
    def get_video_path(self, ann):
        return  pathlib.Path(os.path.join(self.video_root, ann[self.sample_id_key])).resolve()
     
    def get_images_path(self, ann):
        return  pathlib.Path(os.path.join(self.vis_root, ann[self.sample_id_key])).resolve()
    
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):
        raise NotImplementedError("Subclasses should implement this!")
    
    def _build_templates(self, templates_path):
        # use captions not templates
        if templates_path is None:
            self.templates = None
        else:
            with open(templates_path) as f:
                self.templates = json.load(f)

class AudioSetDataset(AudioCaptioningDataset):
    def __init__(self, **kwargs):
        self.dataset_name = 'audioset'
        self.sample_id_key = 'YTID'
        clean_ids = [l.strip() for l in open(kwargs['ann_paths'][-1]).readlines()]
        df = pd.read_csv(kwargs['ann_paths'][-1])
        self.mid2label = {k: v for k, v in zip(df['mid'].tolist(), df['display_name'].tolist())}
        annotation = []
        for ann_path in kwargs['ann_paths'][:-1]:
            df = pd.read_csv(ann_path, comment='#', header=None,names=['YTID', 'start_seconds', 'end_seconds', 'positive_labels'], skiprows=3, quotechar='"', delimiter=',', skipinitialspace=True )
            annotation.extend([row.to_dict() for i,row in df.iterrows()])
        kwargs['ann_paths'] = []
        super().__init__(**kwargs)
        self.annotation = annotation
        self.sample_ids = set.intersection(*[set(getattr(self, f"existing_{modality}_annotation")) for modality in self.modalities])
        
        self.annotation = [ann for ann in self.annotation if ann[self.sample_id_key] in self.sample_ids and ann[self.sample_id_key]]
        self._add_instance_ids()
        print(f"Loaded {len(self.annotation)} examples.")
    
    def get_audio_path(self, ann):
        if 'end_seconds' not in ann:
            ann['start_seconds'] = float(ann['start_time'])
            del ann['start_time']
            ann['end_seconds'] = float(ann['start_seconds']) + 10.0
        return str(os.path.realpath(os.path.join(self.audio_root, ann[self.sample_id_key] + '_{:.1f}-{:.1f}.wav'.format(ann['start_seconds'], ann['end_seconds'])))).replace('all_audio/', '')
    

    def __getitem__(self, index):
        ann = copy.deepcopy(self.annotation[index])
        ann["sample_id"] = ann["YTID"]
        objects = ann['positive_labels'].split(',')
        objects = [self.mid2label[l] for l in objects]
        ann['label'] = objects
        if self.templates:
            ann['captions'] = [random.choice(self.templates).format(obj) for obj in objects]
        else:
            ann['captions'] = [random.choice(objects)]

        for modality in self.modalities:
            ann[f"{modality}_path"] = getattr(self, f"get_{modality}_path")(ann)
            if isinstance(ann[f"{modality}_path"], list):
                ann[f"{modality}_path"] = random.choice(ann[f"{modality}_path"])
            else:
                ann[modality if 'image' not in modality else 'image'] = getattr(self, f"{'vis' if 'image' in modality else modality}_processor")(ann[f"{modality}_path"])
        
        if isinstance(ann['captions'], list):
            ann['text_input'] = self.text_processor(random.choice(ann['captions']))
        else:
            ann['text_input'] = self.text_processor(ann['captions'])

        if ann["audio"].sum() == 0:
            return None

        return ann

class AudioSetInstructDataset(AudioSetDataset):
    def __getitem__(self, index):
        data = super().__getitem__(index)
        if data != None:
            data['text_output'] = data["text_input"]
            data['text_input'] = self.text_processor("")
        return data

class AudioSetEvalDataset(AudioSetDataset):
    def __getitem__(self, index):
        data = super().__getitem__(index)
        if data != None:
            del data["text_input"]
        return data
    
class AudioCapsDataset(AudioCaptioningDataset):
    def __init__(self, **kwargs):
        self.sample_id_key = "youtube_id"
        self.split = 'train' if 'train' in kwargs['ann_paths'][0] else 'test' if 'test' in kwargs['ann_paths'][0] else 'val'
        self.modalities = kwargs['modalities']
        for modality in self.modalities:
            kwargs[f"{modality}_root"] = os.path.join(kwargs[f"{modality}_root"],f'{self.split}')
        super().__init__(**kwargs)
        self.cached = kwargs.get('cached', False)
        self.cache_dir = kwargs.get('cached_dir', '')
        def get_existing_audio_annotations(self):
            return [f.split('_')[0] for f in os.listdir(self.audio_root)] if not self.cached else [f.split('_')[0] for f in os.listdir(self.cached_dir)]

        self.sample_ids = set.intersection(*[set(getattr(self, f"existing_{modality}_annotation")) for modality in self.modalities])
        self.annotation = [ann for ann in self.annotation if ann[self.sample_id_key] in self.sample_ids and ann[self.sample_id_key] not in kwargs.get('missing_ids', [])]
        self._add_instance_ids()
        print(f"Loaded {len(self.annotation)} examples.")
    
    def get_audio_path(self, ann):
        if 'end_seconds' not in ann:
            ann['start_seconds'] = float(ann['start_time'])
            ann['end_seconds'] = ann['start_seconds'] + 10.0
        return os.path.join(self.audio_root, ann[self.sample_id_key] + '_{}.flac'.format(int(ann['start_seconds'])))
    
    def get_cached_audio_path(self, ann):
        if 'end_seconds' not in ann:
            ann['start_seconds'] = float(ann['start_time'])
            ann['end_seconds'] = ann['start_seconds'] + 10.0
        return os.path.join(self.cache_dir, ann[self.sample_id_key] + '_{}.flac.pt'.format(int(ann['start_seconds'])))
    
    def __getitem__(self, index):
        ann = copy.deepcopy(self.annotation[index])
        ann['captions'] = ann['caption']
        ann["sample_id"] = ann["youtube_id"]

        for modality in self.modalities:
            if modality == 'audio' and self.cached:
                ann[f"{modality}_path"] = getattr(self, f"get_cached_{modality}_path")(ann)
                ann["audio"] = torch.load(ann[f"{modality}_path"])
            else:
                ann[f"{modality}_path"] = getattr(self, f"get_{modality}_path")(ann)
                if isinstance(ann[f"{modality}_path"], list):
                    ann[f"{modality}_path"] = random.choice(ann[f"{modality}_path"])
                ann[modality if 'image' not in modality else 'image'] = getattr(self, f"{'vis' if 'image' in modality else modality}_processor")(ann[f"{modality}_path"])
        
        if isinstance(ann['captions'], list):
            ann['text_input'] = self.text_processor(random.choice(ann['captions']))
        else:
            ann['text_input'] = self.text_processor(ann['captions'])

        if ann["audio"].sum() == 0:
            return None

        return ann

class AudioCapsInstructDataset(AudioCapsDataset):
    def __getitem__(self, index):
        data = super().__getitem__(index)
        if data != None:
            data['text_output'] = data["text_input"]
            data['text_input'] = self.text_processor("")
        return data

class AudioCapsEvalDataset(AudioCapsDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        seen = set()
        self.annotation = [x for x in self.annotation if x["youtube_id"] not in seen and not seen.add(x["youtube_id"])]
    
    def __getitem__(self, index):
        data = super().__getitem__(index)
        if data != None:
            del data["text_input"]
        return data

class ClothoV2Dataset(BaseDataset, __DisplMixin):
    def __init__(self, **kwargs):
        super().__init__(kwargs['vis_processor'], kwargs['text_processor'], kwargs['vis_root'], kwargs['ann_paths'])
        # Captions column names in CSV files
        self._CAPTIONS_KEYS = (
            "caption_1",
            "caption_2",
            "caption_3",
            "caption_4",
            "caption_5",
        )
        self.split = kwargs['ann_paths'][-1].split('_')[-1].split('.')[0]
        for ann in self.annotation:
            ann["fname"] = ann["file_name"]
            ann["sound_id"] = ann["fname"]
            ann["captions"] = [ann[caption_key] for caption_key in self._CAPTIONS_KEYS]

        self.audio_processor = kwargs[f"audio_processor"]
        self.audio_root = kwargs[f"audio_root"]
        self._add_instance_ids()

    def __getitem__(self, index):
        ann = copy.deepcopy(self.annotation[index])
        ann['audio'] = self.audio_processor(os.path.join(self.audio_root,self.split,ann['fname']))
        if ann["audio"].sum() == 0:
            return None
        ann['audio_path'] = os.path.join(self.audio_root,self.split,ann['fname'])
        ann["text_input"] = self.text_processor(random.choice(ann['captions']))
        return ann
    
class ClothoV2InstructDataset(ClothoV2Dataset):
    def __getitem__(self, index):
        data = super().__getitem__(index)
        if data != None:
            data['text_output'] = data["text_input"]
            data['text_input'] = self.text_processor("")
        return data

class ClothoV2EvalDataset(ClothoV2Dataset):
    def __getitem__(self, index):
        data = super().__getitem__(index)
        if data != None:
            del data["text_input"]
        return data
    
# class ClothoV2EvalDataset(BaseDataset, __DisplMixin):
#     def __init__(self, **kwargs):
#         super().__init__(kwargs['vis_processor'], kwargs['text_processor'], kwargs['vis_root'], kwargs['ann_paths'])
#         # Captions column names in CSV files
#         self._CAPTIONS_KEYS = (
#             "caption_1",
#             "caption_2",
#             "caption_3",
#             "caption_4",
#             "caption_5",
#         )

#         for ann in self.annotation:
#             ann["fname"] = ann["file_name"]
#             ann["sound_id"] = ann["fname"]
#             ann["captions"] = [ann[caption_key] for caption_key in self._CAPTIONS_KEYS]
#         self.audio_processor = kwargs[f"audio_processor"]
#         self.audio_root = kwargs[f"audio_root"]
#         self._add_instance_ids()

#     def __getitem__(self, index):
#         ann = copy.deepcopy(self.annotation[index])
#         ann['audio'] = self.audio_processor(os.path.join(self.audio_root,ann['fname']))
#         if ann["audio"].sum() == 0:
#             return None
#         ann['audio_path'] = os.path.join(self.audio_root,ann['fname'])
#         # ann["text_input"] = ann['captions']
#         return ann

class AudioLanguagePretrainDataset(BaseDataset, __DisplMixin):
    def __init__(self, **kwargs):
        json_files = kwargs['ann_paths'][:-1]
        blacklist = None
        # self._load_json_file(json_files, kwargs["audio_root"], blacklist)
        self.annotation = json.load(open(kwargs['ann_paths'][-1]))
        self.cached = kwargs.get('cached', False)
        self.cache_dir = kwargs.get('cached_dir', '')
        self.text_processor = kwargs.get('text_processor', None)
        self.audio_processor = kwargs['audio_processor']
        self._add_instance_ids()
    
    # https://github.com/XinhaoMei/WavCaps/blob/c17ff4fe61a650a5d19fb7df8b85569c9ebc74e3/retrieval/data_handling/pretrain_dataset.py#L55
    def _load_json_file(self, files, audio_root, blacklist=None):
        json_data = []
        audio_id = 0
        if blacklist is not None:
            with open(blacklist, 'r') as f:
                blacklist = json.load(f)
        for file in files:
            with open(file, "r") as f:
                json_obj = json.load(f)
                if json_obj["num_captions_per_audio"] == 1:
                    for item in tqdm(json_obj["data"]):
                        if "FreeSound" in file and blacklist is not None:
                            if item["id"] in blacklist["FreeSound"]:
                                continue
                        elif "AudioSet" in file and blacklist is not None:
                            if item["id"] in blacklist["AudioSet"]:
                                continue
                        if 'AudioSet' in file:
                            audio_path = f"{audio_root}/AudioSet_SL_flac/{item['id'].split('.')[0]}.flac"
                        elif 'BBC_Sound' in file:
                            audio_path = f"{audio_root}/BBC_Sound_Effects_flac/{item['id'].split('.')[0]}.flac"
                        elif 'FreeSound' in file:
                            audio_path = f"{audio_root}/FreeSound_flac/{item['id'].split('.')[0]}.flac"
                        elif 'SoundBible' in file:
                            audio_path = f"{audio_root}/SoundBible_flac/{item['id'].split('.')[0]}.flac"
                        if not os.path.exists(audio_path):
                            # print(f'Skipped {audio_path}')
                            continue
                        temp_dict = {"audio": item["audio"], "caption": item["caption"], "id": item['id'],"duration": item["duration"], 'audio_path': audio_path}
                        json_data.append(temp_dict)
                        audio_id += 1
                else:
                    for item in json_obj["data"]:
                        for i in range(1, json_obj["num_captions_per_audio"] + 1):
                            temp_dict = {"audio": item["audio"], "caption": item[f"caption_{i}"], "id": item['id'],
                                        "duration": item["duration"]}
                            json_data.append(temp_dict)
                        audio_id += 1
        return json_data

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        ann = self.annotation[index]

        ## CACHED REPRESENTATIONS
        if self.cached:
            audio = torch.load(os.path.join(self.cache_dir, f"{ann['id']}.pt"), map_location=torch.device('cpu'))
        else:
            audio = self.audio_processor(ann["audio_path"])
        
        if audio.sum() == 0:
            return None

        caption = self.text_processor(ann["caption"])
        audio_id = ann["id"]

        return {
            "audio": audio ,
            "text_input": caption,
            "sample_id": audio_id,
            "instance_id": ann["instance_id"]
        }
    
    def _build_templates(self, templates_path):
        self.templates = None

class AudioLanguagePretrainInstructDataset(AudioLanguagePretrainDataset):
    def __getitem__(self, index):
        data = super().__getitem__(index)
        if data != None:
            data['text_output'] = data["text_input"]
            data['text_input'] = self.text_processor("")
        return data

class AudioLanguagePretrainEvalDataset(AudioLanguagePretrainDataset):
    def __getitem__(self, index):
        data = super().__getitem__(index)
        if data != None:
            del data["text_input"]
        return data