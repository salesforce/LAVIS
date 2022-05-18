import os
import json

from PIL import Image

from datasets.datasets.vqa_datasets import VQADataset, VQAEvalDataset


class COCOVQADataset(VQADataset):
    def __init__(self, vis_processor, text_processor, image_root, ann_paths):
        super().__init__(vis_processor, text_processor, image_root, ann_paths)

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.image_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        question = self.text_processor(ann["question"])

        answer_weight = {}
        for answer in ann["answer"]:
            if answer in answer_weight.keys():
                answer_weight[answer] += 1 / len(ann["answer"])
            else:
                answer_weight[answer] = 1 / len(ann["answer"])

        answers = list(answer_weight.keys())
        weights = list(answer_weight.values())

        return {
            "image": image,
            "question": question,
            "answers": answers,
            "weights": weights,
        }


class COCOVQAEvalDataset(VQAEvalDataset):
    def __init__(self, vis_processor, text_processor, image_root, ann_paths):
        """
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """

        self.image_root = image_root

        self.annotation = json.load(open(ann_paths[0]))
        self.answer_list = json.load(open(ann_paths[1]))

        self.vis_processor = vis_processor
        self.text_processor = text_processor

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.image_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        question = self.text_processor(ann["question"])

        question_id = ann["question_id"]

        return {"image": image, "question": question, "question_id": question_id}
