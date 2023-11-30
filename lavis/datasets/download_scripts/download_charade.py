"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import json
from tqdm import tqdm

train_file = './train.jsonl'
test_file = './test.jsonl'

train_data = [json.loads(l.strip()) for l in open(train_file).readlines()]
test_data = [json.loads(l.strip()) for l in open(test_file).readlines()]

for d in tqdm(train_data):
    d['video_path'] = d['video_id'] + '.mp4'
    d['ts'] = [float(d['start']), float(d['end'])]

for d in tqdm(test_data):
    d['video_path'] = d['video_id'] + '.mp4'
    d['ts'] = [float(d['start']), float(d['end'])]

json.dump(train_data, open('train_lavis.json', 'w'))
json.dump(test_data, open('test_lavis.json', 'w'))