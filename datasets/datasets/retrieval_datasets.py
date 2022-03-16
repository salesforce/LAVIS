import os

from PIL import Image

from datasets.datasets.base_dataset import BaseDataset


class RetrievalDataset(BaseDataset):
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
        
        ann = self.annotation[index]
        
        image_path = os.path.join(self.image_root, ann['image'])        
        image = Image.open(image_path).convert('RGB')   

        image = self.vis_processor(image)
        caption = self.text_processor(ann['caption'])
        
        return image, caption, self.img_ids[ann['image_id']] 


class RetrievalEvalDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, image_root, ann_path):
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        '''
        
        super().__init__(vis_processor, text_processor, image_root, ann_path)
        
        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
        
        txt_id = 0
        for img_id, ann in enumerate(self.annotation):
            self.image.append(ann['image'])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann['caption']):
                self.text.append(self.text_processor(caption))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1

    def __getitem__(self, index):    
        
        image_path = os.path.join(self.image_root, self.annotation[index]['image'])        
        image = Image.open(image_path).convert('RGB')    

        image = self.vis_processor(image)  

        return image, index
    