import os
import json
import random

from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
from torch.utils.data import Dataset
from lavis.datasets.datasets.vqa_datasets import VQADataset

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

        self.dataset = VQADataset(vis_processor, text_processor, vis_root, ann_paths)

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

    def collater(self, samples):
        image_list, question_list, instruction_list, answer_list, weight_list = [], [], [], [], []

        for sample in samples:
            image_list.append(sample["image"])
            question_list.append(sample["text_input"])
            instruction_list.append(sample["instruction"])

            # Sample one answer from the answer list based on weights
            answer_index = random.choices(list(range(len(sample["answers"]))), weights=sample["weights"], k=1)[0]
            answers = sample["answers"][answer_index]
            answer_list.append(answers)
            weight_list.append(sample["weights"][answer_index])

        assert len(question_list) == len(instruction_list)

        text_input = []
        for i in range(len(question_list)):
            text_input.append(instruction_list[i].format(question_list[i]))

        return {
            "image": torch.stack(image_list, dim=0),
            "text_input": text_input,
            "text_output": answer_list,
            "weight": torch.Tensor(weight_list),
        }
