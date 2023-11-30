"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import json
import os

json_path = './violin_annotation.json'

## convert annotations
all_json = json.load(open(json_path))
train_data = [v for v in all_json.values() if 'split' in v and v['split'] == 'train']
test_data = [v for v in all_json.values() if 'split' in v and v['split'] == 'test']

json.dump(train_data, open('train.json', 'w'))
json.dump(test_data, open('test.json', 'w'))