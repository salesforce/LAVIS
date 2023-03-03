from lavis.datasets.datasets.caption_datasets import CaptionDataset
from lavis.datasets.datasets.instruct_wrappers.caption_instruct_datasets import CaptionDatasetInstructWrapper

class COCOCapInstructDataset(CaptionDatasetInstructWrapper):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, instruction_path):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths, instruction_path)
        self.dataset = CaptionDataset(vis_processor, text_processor, vis_root, ann_paths)
