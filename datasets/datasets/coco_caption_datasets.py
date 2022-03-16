import os
import json

from PIL import Image

from datasets.datasets.caption_datasets import CaptionDataset, CaptionEvalDataset

COCOCapDataset = CaptionDataset

class COCOCapEvalDataset(CaptionEvalDataset):
    def __init__(self, vis_processor, text_processor, image_root, ann_path):
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        '''
        super().__init__(vis_processor, text_processor, image_root, ann_path)

    def __getitem__(self, index):    
        ann = self.annotation[index]

        image_path = os.path.join(self.image_root, ann['image'])        
        image = Image.open(image_path).convert('RGB')    

        image = self.vis_processor(image)  

        img_id = ann['image'].split('/')[-1].strip('.jpg').split('_')[-1]

        return image, int(img_id)
