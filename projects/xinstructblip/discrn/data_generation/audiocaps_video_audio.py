"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import random
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm
import pandas as pd
import argparse
from fuzzywuzzy import fuzz
import pickle
import os
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration


parser = argparse.ArgumentParser(description="")
parser.add_argument("--shard", type=int, help="The shard number to process.")
parser.add_argument("--mode", type=str, help=['property', 'get_pairs', 'rtc'])
parser.add_argument("--rnd", type=int, help=["Round of generation"])
parser.add_argument("--split", type=str, help=["train, val"])
parser.add_argument("--original_data_file", type=str)

VIDEO_PATH = '' # f'/export/home/audio_datasets/audiocaps/video/AUDIOCAPS_32000Hz/audio/{split}'

args = parser.parse_args()
shard = args.shard
mode  = args.mode
rnd = args.rnd
split = args.split
original_data_file = args.original_data_file

output_dir = './audio_video_data'
video_list = os.listdir(VIDEO_PATH)
df = pd.read_csv(original_data_file)
# df = pd.read_csv(f'/export/home/audio_datasets/audiocaps/video/AUDIOCAPS_32000Hz/{split}.csv')
df['file_name'] = df['youtube_id'] + "_" + df['start_time'].astype(str) + ".mp4"
df = df[df['file_name'].isin(video_list)]

if mode == 'get_pairs' or mode == 'all':
    sim_model = SentenceTransformer('all-MiniLM-L6-v2')

    def get_pair_by_sim(c1, clist):
        current_embedding = sim_model.encode([c1])  # Shape: [1, D]
        sentences_embeddings = sim_model.encode(clist)  # Shape: [100, D]
        similarities = cosine_similarity(current_embedding, sentences_embeddings)  # Shape: [1, 100]
        sorted_indices = np.argsort(similarities[0])
        top_indices = sorted_indices[-10:]
        sample_idx = np.random.choice(top_indices, size=1, replace=False)[0]
        return sample_idx

if mode != 'get_pairs':
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl", padding='max_length', max_length=512, return_tensors="pt", truncation=True)
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xxl", device_map="auto", torch_dtype=torch.float16)

    def get_output(input_text, input_len=128, output_len=128):
        input_ids = torch.cat([tokenizer(inp, padding='max_length', max_length=input_len, return_tensors="pt").input_ids.to("cuda") for inp in input_text])
        outputs = model.generate(input_ids, max_length=output_len)
        outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return outputs

if mode == 'property' or mode == 'all':
    start_index = shard * (len(df) // 4)
    num_rows_to_extract = len(df) // 4
    df = df.iloc[start_index:start_index + num_rows_to_extract]
    print(f"Subset size {len(df)}")
    captions = df['caption']
    captions = [c.lower() for c in captions]
    properties = []
    bs = 16
    for i in tqdm(range(0,len(captions), bs)):
        input_text = [f"What are three properties to describe an object with description: {c}\n Properties list:" for c in captions[i:i+bs]]
        properties.extend(get_output(input_text, input_len=250, output_len=256))

    df['property'] = properties
    df.to_csv(os.path.join(f'audiocaps_property_shard_{shard}_{rnd}_{split}.csv'))

if mode == 'get_pairs' or mode == "all":
    total_pairs = 100000 if split == 'train' else 10000
    pairs = set() 
    df = pd.read_csv(os.path.join(output_dir, f'audiocaps_property_shard_{shard}_{i}_{split}.csv')).dropna()
    captions = df['caption']
    indices = set(range(len(df)))
    exit_flag = False
    while total_pairs > len(pairs) and not exit_flag:
        for i, row in tqdm(df.iterrows()):
            if len(pairs) == total_pairs:
                break
            curr_caption = row['caption']
            candidates = indices.difference(set([i] + [p[0] for p in pairs if p[1] == i]+ [p[1] for p in pairs if p[0] == i]))
            if len(candidates)>100:
                sample_idx = random.sample(candidates, 500)
            else:
                exit_flag=True
                break
            captions_sample = [captions[i] for i in sample_idx]
            pair_idx = get_pair_by_sim(curr_caption, captions_sample)
            pairs.add(tuple((i, pair_idx)))
            
    pickle.dump(pairs, open(os.path.join(output_dir, f'/audiocaps_pairs_disc_dataset_shard_{shard}_{rnd}_{split}.p', 'wb')))


if mode == "instruction_gen" or mode == "all":
    import pickle
    pairs = pickle.load(open(os.path.join(output_dir, f'audiocaps_pairs_disc_dataset_shard_{shard}_{rnd}_{split}.p', 'rb')))
    
    df = pd.concat([pd.read_csv(os.path.join(output_dir, f'audiocaps_property_shard_{s}_{rnd}_{split}.csv')) for s in [0,1,2,3]])
    examples = []
    prefix = "Given entity A with caption 'A cat meowing and humans speaking on the background.' and corresponding properties: agile, independent and domesticated, \
        and entity B with caption 'Loud barking and traffic' with properties aggressive, loud, and high energy you can generate a set of instruction answer \
            pairs to compare and contrast the entities as follows:\n \
                Examples:\
                Question: Which entity is louder A or B? Answer: Entity B. Explanation: The dogs barking can be heard through traffic making them louder than the cat meowing.\n\
                Question: Which entity shows more social behavior, A or B? Answer: Entity A. Explanation: The cat is surrounded by people whereas the dogs seem to be agitated by traffic.\n\
                Question: Which entity is in higher danger, A or B? Answer: Entity B. Explanation: The dogs seem to be surrounded by traffic which could injure them.\n\
            Generate three such Question, Answer, Explanation tripplets for Entity A with caption {} and properties {} and entity B with caption {} and properties {}\n\
                Examples:\n"
    bs = 16
    start_index = shard * (len(pairs) // 4)
    num_rows_to_extract = len(pairs) // 4
    pairs = list(pairs)
    process_cap = lambda x: x.lower()

    # from pdb import set_trace; set_trace()
    pairs = pairs[start_index:start_index + num_rows_to_extract]
    for i in tqdm(range(0,len(pairs),bs)):
        try:
            input_text = [prefix.format(process_cap(df.iloc[a]['caption']), df.iloc[a]['property'], process_cap(df.iloc[b]['caption']), df.iloc[b]['property']) for a, b in pairs[i:i+bs]]
            outputs = get_output(input_text, input_len=512, output_len=512)
            curr_examples = [{"entity_A":df.iloc[p[0]]['audiocap_id'], "entity_B":df.iloc[p[1]]['audiocap_id'], "caption_A":process_cap(df.iloc[p[0]]['caption']), "caption_B":process_cap(df.iloc[p[1]]['caption']), "property_A":df.iloc[p[0]]['property'], "property_B":df.iloc[p[1]]['property'],"output":outputs[j]} for j,p in enumerate(pairs[i:i+bs])]
            examples.extend(curr_examples)
        except:
            continue
        if i%1000 == 0:
            pickle.dump(examples, open(os.path.join(output_dir, f"disc_examples_{shard}_{rnd}_{split}.p", 'wb')))
    pickle.dump(examples, open(os.path.join(output_dir, f"disc_examples_{shard}_{rnd}_{split}.p", 'wb')))

if mode == "rtc" or mode == "all":
    import pickle
    examples = pickle.load(open(os.path.join(output_dir, f"/disc_examples_{shard}_{rnd}_{split}.p", 'rb')))
    prompt = "Given entity A with caption '{}' and properties {} and entity B with caption '{}' and properties {}. Answer the question: {}. Answer:" 
    rtc_examples = []
    bs = 16
    from pdb import set_trace; set_trace()
    for i in tqdm(range(0,len(examples),bs)):
        try:
            questions =  [e["output"].split("Question:")[1].split("Answer:")[0].strip() for e in examples[i:i+bs]]
            answers = [e["output"].split("Answer:")[1].split("Explanation:")[0].strip() for e in examples[i:i+bs]]
            curr_examples = examples[i:i+bs]
            input_text = [prompt.format(e['caption_A'], e['property_A'],e['caption_B'], e['property_B'], q) for e,q in zip(curr_examples, questions)]
            outputs = get_output(input_text, input_len=512, output_len=30)
            rtc_examples.extend([curr_e for j,curr_e in enumerate(curr_examples) if fuzz.partial_ratio(outputs[j].lower(),answers[j].lower())>90])
        except:
            continue
        if i%1000 == 0:
            pickle.dump(rtc_examples, open(os.path.join(output_dir, f"/disc_examples_rtc_{shard}_{rnd}_{split}.p", 'wb')))
    pickle.dump(rtc_examples, open(os.path.join(output_dir, f"/disc_examples_rtc_{shard}_{rnd}_{split}.p", 'wb')))