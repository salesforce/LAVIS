"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import json 
from tqdm import tqdm
import shutil
import subprocess


image_dir = f'./all_images'
os.makedirs(image_dir, exist_ok=True)
for split in ['train', 'test', 'val']:
    print(f"Processing split {split}...")
    path = f'{os.path.abspath(image_dir)}/{split}/choose_txt'
    annotations = []
    for id in tqdm(os.listdir(path)):
        if not os.path.isdir(os.path.join(path, id)):
            continue
        ann = json.load(open(os.path.join(path, id, 'data.json'), "r"))
        ann['instance_id'] = id
        ann['image_id'] = f'{split}_{id}'
        ann['image'] = f'{split}_{id}.png'
        os.system(' '.join(('ln -s',os.path.join(path, id, 'image.png'),os.path.join(image_dir,ann["image"]))))
        
        annotations.append(ann)
    f = open(f'annotations_{split}.json', 'w')
    f.write(json.dumps(annotations))
    f.close()

    