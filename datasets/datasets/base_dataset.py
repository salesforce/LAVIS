import json

from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __init__(self, vis_processor, text_processor, image_roots, ann_paths):
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        '''        
        # TODO create a merge function to re-structure annotations.
        assert len(set(image_roots)) == 1, "Image roots have to be same for multiple soruce split."

        self.image_root = image_roots[0]

        self.annotation = []
        for ann_path in ann_paths:
            self.annotation.extend(json.load(open(ann_path, 'r')))

        self.vis_processor = vis_processor
        self.text_processor = text_processor

    def __len__(self):
        return len(self.annotation)