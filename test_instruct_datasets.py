"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import lavis.tasks as tasks
from lavis.common.config import Config
from lavis.common.dist_utils import get_rank, init_distributed_mode
from lavis.common.logger import setup_logger
from lavis.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)
from lavis.common.registry import registry
from lavis.common.utils import now

# imports modules for registration
from lavis.datasets.builders import *
from lavis.models import *
from lavis.processors import *
from lavis.runners import *
from lavis.tasks import *
from lavis.datasets.builders import load_dataset
from lavis.models import load_model_and_preprocess

def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--dataset-name", required=True, help="Set which Dataset to test.")
    parser.add_argument("--data-type", default="images", help="[images|videos|features]")
    parser.add_argument("--vis-path", default=None, help="path to the data")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    dataset = load_dataset(args.dataset_name, data_type=args.data_type, vis_path=args.vis_path)
    model, vis_processors, _ = load_model_and_preprocess(
        name="blip2_t5",
        model_type="pretrain_flant5xl",
        is_eval=False,
        device="cpu",
    )

    print("=" * 30)

    for i in range(4):
        print("Text Input:", dataset['train'][i]['text_input'])
        print("Text Output:", dataset['train'][i]['text_output'])

    # Run one sample
    image = vis_processors['train'](dataset['train'][0]['image']).unsqueeze(0).to("cpu")
    loss = model({
        "image": image,
        "text_input": dataset['train'][0]['text_input'],
        "text_output": dataset['train'][0]['text_output'],
    })['loss']
    print('loss:', loss)

    print("=" * 30)
    print('Finished!')


if __name__ == "__main__":
    main()
