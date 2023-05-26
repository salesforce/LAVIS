#!/bin/bash -eu
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

VCR_DIR=/gallery_tate/wonjae.roh
VG_DIR=/gallery_tate/wonjae.roh/vg
Sherlock_DIR=/gallery_tate/wonjae.roh/sherlock_dataset
# "/gallery_tate/wonjae.roh" 까지에 해당하는 부분을 모든 이미지 및 json 파일의 root가 되도록 설정해 주세요
mkdir -p $VG_DIR

wget https://storage.googleapis.com/ai2-mosaic-public/projects/sherlock/data/sherlock_train_v1_1.json.zip -O $Sherlock_DIR/sherlock_train_v1_1.json.zip
wget https://storage.googleapis.com/ai2-mosaic-public/projects/sherlock/data/sherlock_val_with_split_idxs_v1_1.json.zip -O $Sherlock_DIR/sherlock_val_with_split_idxs_v1_1.json.zip
wget https://s3.us-west-2.amazonaws.com/ai2-rowanz/vcr1images.zip -O $VCR_DIR/vcr1images.zip
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip -O $VG_DIR/images.zip
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip -O $VG_DIR/images2.zip

unzip $Sherlock_DIR/sherlock_train_v1_1.json.zip -d $Sherlock_DIR
unzip $Sherlock_DIR/sherlock_val_with_split_idxs_v1_1.json.zip -d $Sherlock_DIR
unzip $VCR_DIR/vcr1images.zip -d $VCR_DIR
unzip $VG_DIR/images.zip -d $VG_DIR/images
unzip $VG_DIR/images2.zip -d $VG_DIR/images