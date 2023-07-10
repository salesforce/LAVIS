import torch
import os

in_dir = "/export/share/dongxuli/zerobooth/500000/blip_model"
out_dir = "/export/share/dongxuli/zerobooth/500000-renamed/blip_model"

checkpoint_filename = "blip_weight.pt"

# 1. load checkpoint
# 2. rename all keys containing "text_model" to "Qformer"
# 3. save checkpoint to out_dir

checkpoint = torch.load(os.path.join(in_dir, checkpoint_filename))
new_checkpoint = {}

for key in checkpoint.keys():
    if "text_model" in key:
        new_key = key.replace("text_model", "Qformer")
    else:
        new_key = key
    new_checkpoint[new_key] = checkpoint[key]

torch.save(new_checkpoint, os.path.join(out_dir, checkpoint_filename))
