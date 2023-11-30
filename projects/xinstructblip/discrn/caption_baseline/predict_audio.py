#!/usr/bin/env python3
# coding: utf-8
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk


# location: WavCaps/captioning
# clone from 
import librosa
import torch
import torch.nn.functional as F
from models.bart_captioning import BartCaptionModel
from lavis.datasets.builders import load_dataset
from tqdm import tqdm
import pickle


## comment out balancind code in dataset before running. also comment out loading data.
ds = load_dataset('audio_video_discrn')['val']

checkpoint_path = "/export/home/WavCaps/Cnn14_Clotho_Spider_31.pt"
audio_path = ""
cp = torch.load(checkpoint_path)
config = cp["config"]
model = BartCaptionModel(config)
model.load_state_dict(cp["model"])
device = torch.device(config["device"])
model.to(device)


entity2pred= {}
for ann in tqdm(ds):
    new_ann = ann
    for i,audio_path in enumerate([ds.get_audio_path(ann, 0), ds.get_audio_path(ann, 1)]):
        if ann['sample_ids'][i] in entity2pred:
            continue
        try:
            waveform, sr = librosa.load(audio_path, sr=32000, mono=True)
            waveform = torch.tensor(waveform)

            if config["audio_encoder_args"]["model_arch"] == "transformer":
                max_length = 32000 * 10
                if len(waveform) > max_length:
                    waveform = waveform[:max_length]
                else:
                    waveform = F.pad(waveform, [0, max_length - len(waveform)], "constant", 0.0)

            else:
                max_length = 32000 * 30
                if len(waveform) > max_length:
                    waveform = waveform[:max_length]

            waveform = waveform.unsqueeze(0)

            model.eval()
            with torch.no_grad():
                waveform = waveform.to(device)
                caption = model.generate(samples=waveform, num_beams=3)
                entity2pred[ann['sample_ids'][i]] = caption
        except:
            print(ann['paths'])
        # print(caption)


pickle.dump(entity2pred, open("./entity2pred/entity2pred_audio.p", 'wb'))
    
