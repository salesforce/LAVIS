import os
import json
import random

from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
from torch.utils.data import Dataset

class VQADatasetInstructWrapper(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, instruction_path):
        """
        TODO:
            Add a probability variable for few-shot examples. E.g.,
            fs_prob = 0.2
            n_fs = k  # the number of examples
        """
        super().__init__()

        with open(instruction_path) as f:
            self.instructions = f.read().splitlines()

        self.dataset = None # VQADataset(vis_processor, text_processor, vis_root, ann_paths)

    def sample_instruction(self):
        instruction = self.instructions[random.randint(0, len(self.instructions) - 1)]
        return instruction

    def process_instruction(self, instruction):
        instruction = instruction.lower()
        instruction = instruction.rstrip("\n")
        instruction = instruction.strip(" ")
        return instruction

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset.__getitem__(index)
        image = data["image"]
        question = data["text_input"]
        answers = data["answers"]
        weights = data["weights"]

        # Sample one answer from the answer list based on weights
        answer_index = random.choices(list(range(len(answers))), weights=weights, k=1)[0]
        answer = answers[answer_index]

        instruction = self.sample_instruction()
        instruction = self.process_instruction(instruction)

        text_input = instruction.format(question)

        return {
            "image": image,
            "text_input": text_input,
            "text_output": answer,
        }
