from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from common.registry import registry
from datasets.builders.base_dataset_builder import BaseDatasetBuilder
from datasets.datasets.retrieval_datasets import RetrievalDataset, RetrievalEvalDataset


# [TODO] to relocate
normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

transform_test = transforms.Compose([
    transforms.Resize((224,224),interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    normalize,
    ])  

transform_train = transforms.Compose([
    transforms.Resize((224,224),interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    normalize,
    ])  

@registry.register_builder("retrieval")
class RetrievalBuilder(BaseDatasetBuilder):
    def __init__(self, cfg):
        super().__init__(cfg)
    
    @classmethod
    def default_config_path(cls):
        raise NotImplementedError
    
    def build(self):
        """
        Create by split datasets inheriting torch.utils.data.Datasets.
        """

        storage_info = self.config.storage
        
        ann_paths = storage_info.get('annotations')
        vis_paths = storage_info.get(self.data_type)

        datasets = dict()
        for split in ann_paths.keys():
            assert split in ['train', 'val', 'test'], "Invalid split name {}, must be one of 'train', 'val' and 'test'."
            is_train = split == 'train'

            dataset_cls = RetrievalDataset if is_train else RetrievalEvalDataset
            transform = transform_train if is_train else transform_test

            datasets[split] = dataset_cls(
                    transform=transform,
                    image_root=vis_paths.get(split),
                    ann_path=ann_paths.get(split)
            )

        return datasets