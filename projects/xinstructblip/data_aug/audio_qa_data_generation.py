 # Copyright (c) 2023, salesforce.com, inc.
 # All rights reserved.
 # SPDX-License-Identifier: BSD-3-Clause
 # For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause


from fuzzywuzzy import fuzz
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm 
import argparse
import pandas as pd
import torch
import os
from transformers import T5Tokenizer, T5ForConditionalGeneration

parser = argparse.ArgumentParser(description="")
parser.add_argument("--mode", type=str, help=['qa_gen', 'rtc'])
parser.add_argument("--split", type=str, help=['val', 'train'])
parser.add_argument("--original_data_file", type=str)

args = parser.parse_args()
split = args.split
mode  = args.mode

original_data_file = #f'/export/home/audio_datasets/audiocaps/video/AUDIOCAPS_32000Hz/{split}.csv'
output_dir = "./audio_qa_data"
os.makedirs(output_dir, exist_ok=True)

## Load Model
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xxl", device_map="auto", torch_dtype=torch.float16)

def get_output(input_text, input_len=128, output_len=128):
    input_ids = torch.cat([tokenizer(inp, padding='max_length', max_length=input_len, return_tensors="pt").input_ids.to("cuda") for inp in input_text])
    outputs = model.generate(input_ids, max_length=output_len)
    outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return outputs

if mode == 'qa_gen':
  df = pd.read_csv(original_data_file)
  print(f"Total number of data: {len(df)}")
  captions = df['caption'].tolist()
  num_examples = len(captions)
  bs = 32
  questions = []
  answers = []
  extractive = []
  for i in tqdm(range(0,num_examples, bs)):
    try:
      input_text = [f"Generate a potential answer word from the following text: {c} " for c in captions[i:i+bs]]
      answers.extend(get_output(input_text, input_len=128, output_len=128))
      input_text = [f"Generate a question for the answer using the context. Context: {c} Answer: {q} Question:" for c,q in zip(captions[i:i+bs], answers[i:i+bs])]
      questions.extend(get_output(input_text, input_len=128, output_len=128))
      extractive.extend([fuzz.partial_ratio(a,c)>90 for c,a in zip(captions[i:i+bs], answers[i:i+bs])])
    except:
      from pdb import set_trace; set_trace()
    
  df['question'] = questions
  df['answer'] = answers
  df['extractive'] = extractive
  print(f'Number extractive: {len([e for e in extractive if e])}')
  df.to_csv(os.path.join(output_dir, f'/audio_qa_{split}.csv'))

if mode == 'rtc':
  df = pd.read_csv(os.path.join(output_dir, f'/audio_qa_{split}.csv'))
  print(f"Total number of data: {len(df)}")
  captions = df['caption'].tolist()
  questions = df['question'].tolist()
  answers = df['answer'].tolist()
  num_examples = len(captions)
  bs = 16
  correct = []
  for i in tqdm(range(0,num_examples, bs)):
    try:
      input_text = [f"Answer the question given the context. Context: {c} Question: {q} Answer:" for c,q in zip(captions[i:i+bs], questions[i:i+bs])]
      outputs = get_output(input_text, input_len=256, output_len=20)
      correct.extend([fuzz.partial_ratio(a,c)>90 for c,a in zip(outputs, answers[i:i+bs])])
    except:
      from pdb import set_trace; set_trace()
    
  df['correct'] = correct
  print(f'Number extractive: {len([e for e in correct if e])}')
  df.to_csv(os.path.join(output_dir, f'/audio_qa_correct_{split}.csv'))
  
  df[df['correct'] == True][df['extractive'] == True].to_csv(os.path.join(output_dir,f'audio_qa_final_{split}.csv'))