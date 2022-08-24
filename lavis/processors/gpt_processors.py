import re

from lavis.common.registry import registry
from lavis.processors.base_processor import BaseProcessor
from lavis.processors.randaugment import RandomAugment
from omegaconf import OmegaConf
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import os 
import pdb 
import numpy as np 
import torch 

class GPTVideoFeatureBaseProcessor(BaseProcessor):
    def __init__(self, visual_ft='i3d_rgb', audio_ft='vggish'):
        self.visual_ft = visual_ft
        self.audio_ft = audio_ft 


@registry.register_processor("gpt_dialogue")
class GPTDialogueProcessor(BaseProcessor):
    def __init__(self, max_turns=10, use_caption=True):
        self.max_turns = max_turns 
        self.use_caption = use_caption 

    def __call__(self, ann):
        if self.use_caption:
            caption = ' '.join([ann['caption'], ann['summary']])
        else:
            caption = ''
            
        dial_history = []
        for turn in ann['dialog'][-self.max_turns:]:
            dial_history.append(turn['question'])
            dial_history.append(turn['answer'])
        dial_history.append(ann['question'])
        answer = ann['answer'] 
        
        return caption, dial_history, answer 

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        use_caption = cfg.get("use_caption", True)
        max_turns = cfg.get("max_turns", 10)

        return cls(max_turns=max_turns, use_caption=use_caption)


@registry.register_processor("gpt_video_ft")
class GPTVideoFeatureProcessor(GPTVideoFeatureBaseProcessor):
    def __init__(
        self, visual_ft, audio_ft
    ):
        super().__init__(visual_ft, audio_ft)
        
        self.transform = transforms.ToTensor()

    def __call__(self, ft_root, vname, ft_type):
        if ft_type == 'visual':
            if self.visual_ft is None: return None 
            ft_path = os.path.join(ft_root, self.visual_ft, vname)
        elif ft_type == 'audio':
            if self.audio_ft is None: return None 
            ft_path = os.path.join(ft_root, self.audio_ft, vname)
        item = np.load(ft_path + '.npy')
        item = self.transform(item) 
        return item 

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()
        
        visual_ft = cfg.get("visual_ft", "i3d_rgb")
        audio_ft = cfg.get("audio_ft", "vggish")
        
        if visual_ft == 'none': visual_ft = None
        if audio_ft == 'none': audio_ft = None

        return cls(
            visual_ft=visual_ft,
            audio_ft=audio_ft
        )


