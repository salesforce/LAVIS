from argparse import Namespace

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import open_clip

from sft_data_utils import make_supervised_data_module

IGNORE_INDEX = -100

if __name__=='__main__':
    # Constant for unit test.
    tokenizer_path = 'lmsys/vicuna-7b-v1.5'
    clip_vision_encoder_path = 'ViT-H-14-378-quickgelu'
    clip_vision_encoder_pretrained = 'dfn5b'
    cache_dir='/export/share/manlis/models'

    # load tokenizer and ensure there is a pad token
    text_tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=False,
        trust_remote_code=True,
        cache_dir=cache_dir,
        use_fast=False, 
    )
    if text_tokenizer.pad_token is None or text_tokenizer.pad_token == text_tokenizer.eos_token:
        # add a pad token if it doesn't exist
        text_tokenizer.add_special_tokens({"pad_token": "<pad>"})

    # add special tokens to the tokenizer and language models
    special_tokens = {
            "media_token": "<image>",
        }
    text_tokenizer.add_special_tokens(
        {"additional_special_tokens": list(special_tokens.values())}
    )

    # load vision encoder
    _, _, image_processor = open_clip.create_model_and_transforms(
        clip_vision_encoder_path,
        pretrained=clip_vision_encoder_pretrained,
        cache_dir=cache_dir,
        force_image_size=378,
    )

    # Create dataset.
    args = Namespace(
        data_sampler_group_by_length=False,
        data_path='/export/share/manlis/data/lavis/llava_instruct_665k_sharegpt4v/annotations/sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.json',
        batch_size=8,
        world_size=8,
        gradient_accumulation_steps=1,
        rank=0,
        workers=4,
        image_aspect_ratio='pad',
        is_multimodal=True,
        mm_use_im_start_end=False,
    )
    train_dataset, total_num_samples = make_supervised_data_module(tokenizer=text_tokenizer, 
                                                                   image_processor=image_processor, 
                                                                   data_args=args)
    # Iter through all data samples.
    print(len(train_dataset.dataloader))
    for i, sample in enumerate(train_dataset.dataloader):
        if (sample['labels'] == IGNORE_INDEX).all():
            print(f"sample {i} token mismatch")
        pass
