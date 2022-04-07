import torch

from datasets.datasets.base_dataset import BaseDataset


class VQADataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, image_roots, ann_paths):
        super().__init__(vis_processor, text_processor, image_roots, ann_paths)

    def collater(self, samples):
        image_list, question_list, answer_list, weight_list = [], [], [], []

        num_answers = []

        for sample in samples:
            image_list.append(sample["image"])
            question_list.append(sample["question"])

            weight_list.extend(sample["weights"])

            answers = sample["answers"]

            answer_list.extend(answers)
            num_answers.append(len(answers))

        return {
            "image": torch.stack(image_list, dim=0),
            "question": question_list,
            "answer": answer_list,
            "weight": torch.Tensor(weight_list),
            "n_answers": torch.LongTensor(num_answers),
        }


class VQAEvalDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, image_roots, ann_paths):
        super().__init__(vis_processor, text_processor, image_roots, ann_paths)
