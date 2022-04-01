import os

from PIL import Image


from datasets.datasets.base_dataset import BaseDataset

class VQADataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, image_roots, ann_paths):
        super().__init__(vis_processor, text_processor, image_roots, ann_paths)
    
    def __getitem__(self, index):
        ann = self.annotation[index]
        
        image_path = os.path.join(self.image_root, ann['image'])
        image = Image.open(image_path).convert('RGB')

        image = self.vis_processor(image)  
        question = self.text_processor(ann['question'])

        # question = pre_question(ann['question'])        
        
        answer_weight = {}
        for answer in ann['answer']:
            if answer in answer_weight.keys():
                answer_weight[answer] += 1/len(ann['answer'])
            else:
                answer_weight[answer] = 1/len(ann['answer'])

        answers = list(answer_weight.keys())
        weights = list(answer_weight.values())

        return image, question, answers, weights


class VQAEvalDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, image_roots, ann_paths):
        super().__init__(vis_processor, text_processor, image_roots, ann_paths)
        
    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.image_root, ann['image'])
        image = Image.open(image_path).convert('RGB')

        image = self.vis_processor(image)  
        question = self.text_processor(ann['question'])

        # image = self.vis_processor(image)  

        # question = pre_question(ann['question'])   
        question_id = ann['question_id']            

        return image, question, question_id