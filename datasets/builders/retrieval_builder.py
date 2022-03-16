import os

from common.registry import registry
from datasets.builders.base_dataset_builder import BaseDatasetBuilder
from datasets.datasets.retrieval_datasets import RetrievalDataset, RetrievalEvalDataset


@registry.register_builder("retrieval")
class RetrievalBuilder(BaseDatasetBuilder):
    def __init__(self, cfg):
        super().__init__(cfg)
    
    @classmethod
    def default_config_path(cls):
        raise NotImplementedError
    
    def build_processors(self):
        """
        Building processors is task-specific.

        Some tasks require both visual and text processors, e.g. retrieval, QA.
        Some tasks require only visual processors, e.g. captioning.
        """
        self.vis_processors['train'] = registry.get_processor_class(self.config.vis_processor['train'])()
        self.vis_processors['eval'] = registry.get_processor_class(self.config.vis_processor['eval'])()

        self.text_processors['train'] = registry.get_processor_class(self.config.text_processor['train'])()
        self.text_processors['eval'] = registry.get_processor_class(self.config.text_processor['eval'])()

    def build(self):
        """
        Create by split datasets inheriting torch.utils.data.Datasets.
        """
        self.build_processors()

        storage_info = self.config.storage
        
        ann_paths = storage_info.get('annotations')
        vis_paths = storage_info.get(self.data_type)

        datasets = dict()
        for split in ann_paths.keys():
            assert split in ['train', 'val', 'test'], "Invalid split name {}, must be one of 'train', 'val' and 'test'."
            is_train = split == 'train'

            # processors
            vis_processor = self.vis_processors['train'] if is_train else self.vis_processors['eval']
            text_processor = self.text_processors['train'] if is_train else self.text_processors['eval']

            # annotation path
            ann_path = ann_paths.get(split)
            if not os.path.isabs(ann_path):
                ann_path = os.path.join(registry.get_path("cache_root"), ann_path) 
            
            vis_path = vis_paths.get(split)
            if not os.path.isabs(vis_path):
                vis_path = os.path.join(registry.get_path("cache_root"), vis_path) 

            # create datasets
            dataset_cls = RetrievalDataset if is_train else RetrievalEvalDataset
            datasets[split] = dataset_cls(
                    vis_processor=vis_processor,
                    text_processor=text_processor,
                    ann_path=ann_path,
                    image_root=vis_path
                )

        return datasets