import os
import json

from PIL import Image

from datasets.datasets.base_dataset import BaseDataset


class CaptionDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, image_root, ann_path):
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        '''        
        super().__init__(vis_processor, text_processor, image_root, ann_path)

        self.img_ids = {}  
        n = 0
        for ann in self.annotation:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1    
        
    def __getitem__(self, index):    
        
        # TODO this assumes image input, not general enough
        ann = self.annotation[index]
        
        image_path = os.path.join(self.image_root, ann['image'])        
        image = Image.open(image_path).convert('RGB')   

        image = self.vis_processor(image)
        caption = self.text_processor(ann['caption'])

        return {"vis_data": image, "text_data": caption, "image_id": self.img_ids[ann["image_id"]]}


class CaptionEvalDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, image_root, ann_path):
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        '''
        super().__init__(vis_processor, text_processor, image_root, ann_path)

    def __getitem__(self, index):    
        
        image_path = os.path.join(self.image_root, self.annotation[index]['image'])        
        image = Image.open(image_path).convert('RGB')    

        image = self.vis_processor(image)  

        return {"vis_data": image, "image_id": self.annotations[index]["image_id"]}
    