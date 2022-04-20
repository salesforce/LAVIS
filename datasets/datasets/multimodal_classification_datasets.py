from datasets.datasets.base_dataset import BaseDataset


class MultimodalClassificationDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, image_roots, ann_paths):
        super().__init__(vis_processor, text_processor, image_roots, ann_paths)

        self.class_labels = None
