from abc import abstractmethod
from lavis.datasets.datasets.base_dataset import BaseDataset


class MultimodalClassificationDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.class_labels = None

    @abstractmethod
    def _build_class_labels(self):
        pass
