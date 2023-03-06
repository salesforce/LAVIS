from lavis.datasets.datasets.instruct_wrappers.vqa_instruct_datasets import VQADatasetInstructWrapper
from lavis.datasets.datasets.snli_ve_datasets import SNLIVisualEntialmentDataset


class SNLIVisualEntialmentInstructDataset(VQADatasetInstructWrapper):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, instruction_path):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths, instruction_path)
        self.dataset = SNLIVisualEntialmentDataset(vis_processor, text_processor, vis_root, ann_paths)

        self.label2class = {0: "contradiction", 1: "neutral", 2: "entailment"}

    def __getitem__(self, index):
        data = self.dataset.__getitem__(index)

        instruction = self.sample_instruction()
        instruction = self.process_instruction(instruction)

        text_input = instruction.format(data["text_input"])

        return {
            "image": data["image"],
            "text_input": text_input,
            "text_output": self.label2class[data["label"]],
        }