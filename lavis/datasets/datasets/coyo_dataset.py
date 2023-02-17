"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import webdataset as wds
from lavis.datasets.datasets.base_dataset import BaseDataset
import random
from glob import glob
import os

class CoyoDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, location):
        super().__init__(vis_processor=vis_processor, text_processor=text_processor)

        filelist = glob(os.path.join(location,'*','*.tar'))
        self.inner_dataset = wds.DataPipeline(
            wds.ResampledShards(filelist),
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
            wds.shuffle(1000, handler=wds.warn_and_continue),
            wds.decode('pilrgb', handler=wds.warn_and_continue),
            wds.to_tuple('jpg', 'json', handler=wds.warn_and_continue),
            wds.map_tuple(self.vis_processor, handler=wds.warn_and_continue),
            wds.map(self.to_dict, handler=wds.warn_and_continue),
        )
    def to_dict(self, sample):
        metadata = sample[1]
        clip_scores = [metadata["caption_clipscore"],metadata["caption_blip2_clipscore"],metadata["caption_blip2_coco_clipscore"]]
        candidates = [metadata['caption'],metadata['caption_blip2'],metadata['caption_blip2_coco']]
        captions = [c for (c,s) in zip(candidates,clip_scores) if s>0.1] #filter out noisy captions
        if not captions:
            captions = candidates #randomly select one caption if all are considered noisy
        caption = random.choice(captions)            
        return {
            "image": sample[0],
            "text_input": caption,
        }


if __name__ == "__main__":
    from torchvision import transforms

    normalize = transforms.Normalize(
        (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
    )

    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(256, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )

    dataset = CoyoDataset(
        vis_processor=transform_train,
        text_processor=lambda x: x,
        location="/export/coyo700m/caption_blip2_coco",
    )

    import torch

    loader = torch.utils.data.DataLoader(dataset.inner_dataset, batch_size=2)

    print(next(iter(loader))["text_input"])
