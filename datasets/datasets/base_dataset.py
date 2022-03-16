import json

from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __init__(self, vis_processor, text_processor, image_root, ann_path):
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        '''        

        self.image_root = image_root
        self.annotation = json.load(open(ann_path,'r'))

        self.vis_processor = vis_processor
        self.text_processor = text_processor

    def __len__(self):
        return len(self.annotation)