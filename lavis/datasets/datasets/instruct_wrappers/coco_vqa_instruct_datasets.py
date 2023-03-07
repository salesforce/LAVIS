from lavis.datasets.datasets.instruct_wrappers.vqa_instruct_datasets import VQADatasetInstructWrapper
from lavis.datasets.datasets.coco_vqa_datasets import COCOVQADataset


class COCOVQAInstructDataset(VQADatasetInstructWrapper):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, instruction_path):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths, instruction_path)
        self.dataset = COCOVQADataset(vis_processor, text_processor, vis_root, ann_paths)
