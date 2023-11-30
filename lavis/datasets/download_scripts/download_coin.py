"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""


## Pre-requisities: run 'pip install youtube-dl' to install the youtube-dl package.
## Specify your location of output videos and input json file.
## It can also be used for youcookii by updating the file paths.
import json
import os

output_path = './videos'
json_path = './COIN.json'

if not os.path.exists(output_path):
	os.mkdir(output_path)
	
data = json.load(open(json_path, 'r'))['database']
youtube_ids = list(data.keys())

for youtube_id in data:
	info = data[youtube_id]
	type = info['recipe_type']
	url = info['video_url']
	vid_loc = output_path + '/' + str(type)
	if not os.path.exists(vid_loc):
		os.mkdir(vid_loc)
	os.system('youtube-dl -o ' + vid_loc + '/' + youtube_id + '.mp4' + ' -f best ' + url)
	
	# To save disk space, you could download the best format available 
	# 	but not better that 480p or any other qualities optinally
	# See https://askubuntu.com/questions/486297/how-to-select-video-quality-from-youtube-dl

## convert annotations
all_json = json.load(open(json_path))['database']
train_data = []
test_data = []
for k,v in all_json.items():
	for gt_ann in v['annotation']:
		new_ann = {}
		youtube_id = v["video_url"].split("/")[-1]
		new_ann['youtube_id'] = youtube_id
		new_ann["recipe_type"] = v["recipe_type"]
		new_ann['video_path'] = f'{v["recipe_type"]}/{youtube_id}.mp4'
		new_ann['caption'] = gt_ann['label']
		new_ann['id'] = gt_ann['id']
		new_ann['ts'] = gt_ann['ts']
		if v['subset'] == 'training':
			train_data.append(new_ann)
		else:
			test_data.append(new_ann)
	
json.dump(train_data, open('train.json', 'w'))
json.dump(test_data, open('test.json', 'w'))