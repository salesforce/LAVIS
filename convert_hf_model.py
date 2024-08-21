import torch
import os
from pathlib import Path
import argparse
from omegaconf import OmegaConf
import torch
from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoImageProcessor

from open_flamingo import create_model_and_transforms


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dest_fn",
        type=str,
        default="./base_model_weight/xgen-mm-phi3-mini-base-r-v1.5.pt",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    # Load model from HF hub.
    model_name_or_path = "Salesforce/xgen-mm-phi3-mini-base-r-v1.5"
    model = AutoModelForVision2Seq.from_pretrained(
        model_name_or_path, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=True, use_fast=True, legacy=False
    )
    image_processor = AutoImageProcessor.from_pretrained(
        model_name_or_path, trust_remote_code=True
    )
    tokenizer = model.update_special_tokens(tokenizer)

    # Test weight loading.
    # Set local model configs.
    cfg = dict(
        model_family="xgenmm_v1",
        lm_path="microsoft/Phi-3-mini-4k-instruct",
        vision_encoder_path="google/siglip-so400m-patch14-384",
        vision_encoder_pretrained="google",
        num_vision_tokens=128,
        image_aspect_ratio="anyres",
        anyres_patch_sampling=True,
        anyres_grids=[(1, 2), (2, 1), (2, 2), (3, 1), (1, 3)],
    )
    cfg = OmegaConf.create(cfg)

    additional_kwargs = {
        "num_vision_tokens": cfg.num_vision_tokens,
        "image_aspect_ratio": cfg.image_aspect_ratio,
        "anyres_patch_sampling": cfg.anyres_patch_sampling,
    }

    # Initialize the model.
    local_model, _, _ = create_model_and_transforms(
        clip_vision_encoder_path=cfg.vision_encoder_path,
        clip_vision_encoder_pretrained=cfg.vision_encoder_pretrained,
        lang_model_path=cfg.lm_path,
        tokenizer_path=cfg.lm_path,
        model_family=cfg.model_family,
        **additional_kwargs,
    )

    try:
        local_model.load_state_dict(model.vlm.state_dict(), strict=True)
        print("Testing weight loading OK.")
    except Exception as e:
        print(e)

    # Export model weight.
    print(f"Saving converted model weight to {args.dest_fn}")
    Path(args.dest_fn).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.vlm.state_dict(), args.dest_fn)
