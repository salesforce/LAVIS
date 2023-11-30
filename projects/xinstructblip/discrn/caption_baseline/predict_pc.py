 # Copyright (c) 2023, salesforce.com, inc.
 # All rights reserved.
 # SPDX-License-Identifier: BSD-3-Clause
 # For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause


import torch
import torch.nn.functional as F
from lavis.datasets.builders import load_dataset
from tqdm import tqdm

from lavis.models import load_model_and_preprocess
import argparse
import json
from PIL import Image
import os
import random
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision


def check_positive(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return ivalue

def check_float_range(min_val, max_val):
    def helper(x):
        x = float(x)
        if x < min_val or x > max_val:
            raise argparse.ArgumentTypeError("Value must be between {} and {}.".format(min_val, max_val))
        return x
    return helper

def setup_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True

# Create the top-level parser
parser = argparse.ArgumentParser(
    description='A demo script for using instructBLIP models for tasks such as image captioning or Visual Question Answering (VQA).'
)

# Model Name
parser.add_argument(
    '--model-name',
    default='blip2_vicuna_instruct',
    choices=['blip2_vicuna_instruct', 'blip2_t5_instruct'],
    help='The name of the instructBLIP model to use.'
)

# Model Type
parser.add_argument(
    '--model-type',
    default='vicuna7b',
    choices=['vicuna7b', 'vicuna13b', 'flant5xl', 'flant5xxl'],
    help='The type of the model to use.'
)

# Task
parser.add_argument(
    '--task',
    default='caption',
    choices=['caption', 'vqa'],
    help='The type of task to run: captioning or VQA.'
)

# Number of tasks
parser.add_argument(
    '--num',
    type=int,
    default=1,
    help='The number of outputs to generate.'
)

# GPU ID
parser.add_argument(
    '--gpu-id',
    type=int,
    default=0,
    help='The ID of the GPU to use.'
)

# Image path or URL
parser.add_argument(
    '--image_path_or_url',
    type=str,
    default='',
    help='The path or URL to the image file.'
)

# Prompt
parser.add_argument(
    '--prompt',
    type=str,
    default='',
    help='The text prompt to use in tasks. Enclose it in quotation ("")'
)

# Seed
parser.add_argument(
    '--seed',
    type=int,
    default=42,
    help='The seed for the random number generator.'
)

# Minimum Length
parser.add_argument(
    '--min_len',
    type=check_positive,
    default=1,
    help='The minimum length for the generated text.'
)

# Maximum Length
parser.add_argument(
    '--max_len',
    type=check_positive,
    default=250,
    help='The maximum length for the generated text.'
)

# Beam Size
parser.add_argument(
    '--beam_size',
    type=check_positive,
    default=5,
    help='The beam size for the Beam Search.'
)

# Length Penalty
parser.add_argument(
    '--len_penalty',
    type=float,
    default=-1,
    help='The length penalty for Beam Search.'
)

# Repetition Penalty
parser.add_argument(
    '--rep_penalty',
    type=float,
    default=1,
    help='The penalty for word repetitions in the generated text.'
)

# Top P
parser.add_argument(
    '--top_p',
    type=check_float_range(0.0, 1.0),
    default=0.9,
    help='The cumulative probability threshold for Nucleus Sampling.'
)

# Decoding Method
parser.add_argument(
    '--decoding_method',
    default='Nucleus sampling',
    choices=['Nucleus sampling', 'Beam search'],
    help='The method to use for decoding the generated text.'
)

# Temperature
parser.add_argument(
    '--temperature',
    type=check_float_range(0.1, 5.0),
    default=1.0,
    help='The temperature to use in the generation process. Higher values increase randomness, lower values make the output more deterministic.'
)

args = parser.parse_args()

if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu_id)
    device='cuda'
    print(f"Script will run on cuda device {args.gpu_id}")
else:
    print("CUDA is not available. The script will run on CPU.")
    device='cpu'


print(f"Setting up seeds [seed={args.seed}]")
setup_seeds(args.seed)

print(f'Loading model {args.model_name} with LLM {args.model_type}...')
model, vis_processors, _ = load_model_and_preprocess(
    name=args.model_name,
    model_type=args.model_type,
    is_eval=True,
    device=device,
)

ds = load_dataset('image_pc_discrn')
vis_processors = ds['val'].vis_processor
print('Loading model done!')

def inference(image, prompt, min_len=1, max_len=250, beam_size=5, len_penalty=-1, repetition_penalty=1, top_p=.9, decoding_method='Beam Search',num_captions=1, temperature=1., video=False):
    use_nucleus_sampling = decoding_method == "Nucleus sampling"
    print(image, prompt, min_len, max_len, beam_size, len_penalty, repetition_penalty, top_p, use_nucleus_sampling)
    if not video:
        image = torchvision.transforms.functional.rgb_to_grayscale(Image.open(image), 3)
        image = vis_processors(image).unsqueeze(0).to(device)
    else:
        image = vis_processors(image).to(device).unsqueeze(0).half()
        # image = torch.cat(processed_frames, dim=0).mean(dim=0, keepdim=True).to(device)

    samples = {
        "image": image,
        "prompt": prompt,
    }

    output = model.generate(
        samples,
        repetition_penalty=float(repetition_penalty),
        num_beams=beam_size,
        max_length=max_len,
        min_length=min_len,
        top_p=top_p,
        use_nucleus_sampling=use_nucleus_sampling,
        num_captions=num_captions,
        temperature=temperature,
    )
    return output[0]


import pickle
## comment out balancind code in dataset before running. also comment out loading data.
# ds = load_dataset('image_pc_discrn')['val']
pc_path = "/export/einstein-vision/3d_vision/render_pc_discrn"
entity2pred= {}
for ann in tqdm(os.listdir(pc_path)):
        if '8192' in ann:
            with torch.no_grad():
                sample_id = ann.split('_')[0]
                caption = inference(os.path.join(pc_path,ann), "describe the image", video=False)
                entity2pred[sample_id] = caption
            # except:
            #     print(ann)
        # print(caption)


pickle.dump(entity2pred, open("./entity2pred/entity2pred_pc.p", 'wb'))