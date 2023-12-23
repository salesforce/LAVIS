 # Copyright (c) 2023, salesforce.com, inc.
 # All rights reserved.
 # SPDX-License-Identifier: BSD-3-Clause
 # For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

from tqdm import tqdm 
import argparse
import pandas as pd
import torch
import os
from transformers import T5Tokenizer, T5ForConditionalGeneration
from fuzzywuzzy import fuzz

parser = argparse.ArgumentParser(description="")
parser.add_argument("--shard", type=int, help="The shard number to process.")
parser.add_argument("--mode", type=str, help=['color_removal', 'qa_gen', 'rtc'])
parser.add_argument("--split", type=str, help=['train', 'val'])
parser.add_argument("--original_data_file", type=str, help=['Download csv file from https://huggingface.co/datasets/tiange/Cap3D/blob/main/Cap3D_automated_Objaverse_no3Dword.csv'])

args = parser.parse_args()
shard = args.shard
mode  = args.mode
split = args.split
original_data_file = args.original_datafile
# original_data_file = f'/export/einstein-vision/3d_vision/objaverse_captions/objaverse_blip_captions_no3d_{split}.csv'
output_dir = "./3d_qa_data"
os.makedirs(output_dir, exist_ok=True)

## Load Model
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xxl", device_map="auto", torch_dtype=torch.float16)

def get_output(input_text, input_len=128, output_len=128):
    input_ids = torch.cat([tokenizer(inp, padding='max_length', max_length=input_len, return_tensors="pt").input_ids.to("cuda") for inp in input_text])
    outputs = model.generate(input_ids, max_length=output_len)
    outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return outputs
  

if mode == 'color_removal' or mode == 'all':
  df = pd.read_csv(original_data_file, names=["sample_id", "caption"])
  print(f"Total captions: {len(df)}")
  start_index = shard * (len(df) // 4)
  num_rows_to_extract = len(df) // 4
  df = df.iloc[start_index:start_index + num_rows_to_extract]
  ## remove color. 
  no_color_captions = []
  captions = df["caption"].tolist()
  num_examples = len(captions)
  bs = 64
  for i in tqdm(range(0,num_examples, bs)):
    input_text = [f"Rewrite the sentence {c} by removing mentions of color." for c in captions[i:i+bs]]
    no_color_captions.extend(get_output(input_text, input_len=128, output_len=128))

  df['caption_no_color'] = no_color_captions
  df.to_csv(os.path.join(output_dir,f'Cap3D_automated_Objaverse_no_color_shard_{shard}_{split}.csv'))

if mode == 'qa_gen' or mode == 'all':
  df = pd.read_csv(os.path.join(output_dir,f'/Cap3D_automated_Objaverse_no_color_shard_{shard}_{split}.csv')).dropna()
  df = df[df['caption_no_color'].apply(lambda x: len(str(x).split(' ')) > 10)]
  print(f"Total number of data: {len(df)}")
  captions = df['caption_no_color'].tolist()
  num_examples = len(captions)
  bs = 32
  questions = []
  answers = []
  extractive = []
  for i in tqdm(range(0,num_examples, bs)):
    try:
      input_text = [f"Generate a potential answer word from the following text: {c} " for c in captions[i:i+bs]]
      answers.extend(get_output(input_text, input_len=180, output_len=128))
      input_text = [f"Generate a question for the answer using the context. Context: {c} Answer: {q} Question:" for c,q in zip(captions[i:i+bs], answers[i:i+bs])]
      questions.extend(get_output(input_text, input_len=180, output_len=30))
      extractive.extend([fuzz.partial_ratio(a,c)>90 for c,a in zip(captions[i:i+bs], answers[i:i+bs])])
    except:
      from pdb import set_trace; set_trace()
    
  df['question'] = questions
  df['answer'] = answers
  df['extractive'] = extractive
  print(f'Number extractive: {len([e for e in extractive if e])}')
  df.to_csv(os.path.join(output_dir,f'/Cap3D_automated_Objaverse_no_color_qa_shard_{shard}_{split}.csv'))

if mode == 'rtc' or mode == 'all':
  df = pd.read_csv(os.path.join(output_dir, f'Cap3D_automated_Objaverse_no_color_qa_shard_{shard}_{split}.csv')).dropna()
  print(f"Total number of data: {len(df)}")
  captions = df['caption_no_color'].tolist()
  num_examples = len(captions)
  bs = 32
  questions = df['question'].tolist()
  answers =df['answer'].tolist()
  correct = []
  for i in tqdm(range(0,num_examples, bs)):
    try:
      input_text = [f"Answer the question given the context. Context: {c} Question: {q} Answer:" for c,q in zip(captions[i:i+bs], questions[i:i+bs])]
      outputs = get_output(input_text, input_len=256, output_len=20)
      correct.extend([fuzz.partial_ratio(a,c)>90 for c,a in zip(outputs, answers[i:i+bs])])
    except:
      from pdb import set_trace; set_trace()
    
  df['correct'] = correct
  print(f'Number correct: {len([e for e in correct if e])}')
  df.to_csv(os.path.join(output_dir, f'/Cap3D_automated_Objaverse_no_color_qa_correct_shard_{shard}_{split}.csv'))
  
  df[df['extractive'] == True][df['correct'] == True].to_csv(os.path.join(output_dir, f'/CAP3DQA_final_shard_{shard}_{split}.csv'))