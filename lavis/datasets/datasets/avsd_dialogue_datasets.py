import os
from lavis.datasets.datasets.base_dataset import BaseDataset

from lavis.datasets.datasets.dialogue_datasets import DialogueDataset, DialogueEvalDataset

import pdb 

class AVSDDialDataset(DialogueDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):

        ann = self.annotation[index]

        vname = ann["image_id"]
        #video_path = os.path.join(self.vis_root, vname)

        video_visual = self.vis_processor(self.vis_root, vname, 'visual')
        video_audio = self.vis_processor(self.vis_root, vname, 'audio')
        
        caption, dial_history, answer = self.text_processor(ann)
        
        # "image_id" is kept to stay compatible with the COCO evaluation format
        return {
            "video_visual": video_visual,
            "video_audio": video_audio, 
            "caption": caption, 
            "dial_history": dial_history,
            "answer": answer, 
            "image_id": ann["image_id"],
            "instance_id": ann["instance_id"]
        }


class AVSDDialEvalDataset(DialogueEvalDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):

        ann = self.annotation[index]

        vname = ann["video"]
        video_path = os.path.join(self.vis_root, vname)

        video = self.vis_processor(video_path)

        return {
            "video": video,
            "image_id": ann["image_id"],
            "instance_id": ann["instance_id"],
        }
