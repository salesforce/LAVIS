from lavis.datasets.datasets.instruct_wrappers.vqa_instruct_datasets import VQADatasetInstructWrapper
from lavis.datasets.datasets.coco_vqa_datasets import COCOVQADataset
# COCOVQAInstructDataset = VQADatasetInstructWrapper

class COCOVQAInstructDataset(VQADatasetInstructWrapper):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, instruction_path):
        """
        TODO:
            Add a probability variable for few-shot examples. E.g.,
            fs_prob = 0.2
            n_fs = k  # the number of examples
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths, instruction_path)

        self.dataset = COCOVQADataset(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):
        data = self.dataset.__getitem__(index)

        instruction = self.sample_instruction()
        instruction = self.process_instruction(instruction)

        data["instruction"] = instruction
        return data
