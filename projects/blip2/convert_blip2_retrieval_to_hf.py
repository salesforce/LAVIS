"""
Convert a trained LAVIS BLIP-2 retrieval checkpoint to Hugging Face format.

The generated folder can be loaded with
`transformers.Blip2ForImageTextRetrieval.from_pretrained`.
"""

import argparse
import importlib.machinery
import logging
from pathlib import Path
import sys
import types
from urllib.parse import urlparse


LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def make_stub_module(name):
    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    return module


def install_lavis_optional_dependency_stubs():
    """Avoid importing optional video/3D dependencies for an image-only converter."""
    try:
        from transformers.models.clip import modeling_clip

        if not hasattr(modeling_clip, "_expand_mask"):

            def _expand_mask(mask, dtype, tgt_len=None):
                import torch

                batch_size, src_len = mask.size()
                tgt_len = tgt_len if tgt_len is not None else src_len
                expanded_mask = mask[:, None, None, :].expand(
                    batch_size, 1, tgt_len, src_len
                )
                inverted_mask = 1.0 - expanded_mask.to(dtype)
                return inverted_mask.masked_fill(
                    inverted_mask.to(torch.bool), torch.finfo(dtype).min
                )

            modeling_clip._expand_mask = _expand_mask
    except ImportError:
        pass

    if "decord" not in sys.modules:
        decord_stub = make_stub_module("decord")

        class VideoReader:
            def __init__(self, *args, **kwargs):
                raise ImportError("decord is required for video datasets.")

        def cpu(*args, **kwargs):
            return None

        decord_stub.VideoReader = VideoReader
        decord_stub.cpu = cpu
        decord_stub.bridge = types.SimpleNamespace(set_bridge=lambda *args, **kwargs: None)
        sys.modules["decord"] = decord_stub

    if "open3d" not in sys.modules:
        sys.modules["open3d"] = make_stub_module("open3d")

    if "torchaudio" not in sys.modules:
        torchaudio_stub = make_stub_module("torchaudio")
        transforms_stub = make_stub_module("torchaudio.transforms")
        compliance_stub = make_stub_module("torchaudio.compliance")
        kaldi_stub = make_stub_module("torchaudio.compliance.kaldi")

        def load(*args, **kwargs):
            raise ImportError("torchaudio is required for audio datasets.")

        class Resample:
            def __init__(self, *args, **kwargs):
                raise ImportError("torchaudio is required for audio datasets.")

        def fbank(*args, **kwargs):
            raise ImportError("torchaudio is required for audio datasets.")

        torchaudio_stub.load = load
        transforms_stub.Resample = Resample
        kaldi_stub.fbank = fbank
        compliance_stub.kaldi = kaldi_stub
        torchaudio_stub.transforms = transforms_stub
        torchaudio_stub.compliance = compliance_stub

        sys.modules["torchaudio"] = torchaudio_stub
        sys.modules["torchaudio.transforms"] = transforms_stub
        sys.modules["torchaudio.compliance"] = compliance_stub
        sys.modules["torchaudio.compliance.kaldi"] = kaldi_stub

    if "moviepy.editor" not in sys.modules:
        moviepy_stub = sys.modules.get("moviepy") or make_stub_module("moviepy")
        editor_stub = make_stub_module("moviepy.editor")

        class VideoFileClip:
            def __init__(self, *args, **kwargs):
                raise ImportError("moviepy is required for audio/video datasets.")

        editor_stub.VideoFileClip = VideoFileClip
        moviepy_stub.editor = editor_stub
        sys.modules["moviepy"] = moviepy_stub
        sys.modules["moviepy.editor"] = editor_stub

    if "diffusers" not in sys.modules:
        diffusers_stub = make_stub_module("diffusers")
        diffusers_stub.__path__ = []
        models_stub = make_stub_module("diffusers.models")
        models_stub.__path__ = []
        cross_attention_stub = make_stub_module("diffusers.models.cross_attention")
        utils_stub = make_stub_module("diffusers.utils")
        utils_stub.__path__ = []
        pil_utils_stub = make_stub_module("diffusers.utils.pil_utils")

        class DiffusersStub:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                raise ImportError("diffusers is required for BLIP-Diffusion models.")

        for name in (
            "AutoencoderKL",
            "ControlNetModel",
            "DDPMScheduler",
            "DDIMScheduler",
            "PNDMScheduler",
            "UNet2DConditionModel",
        ):
            setattr(diffusers_stub, name, type(name, (DiffusersStub,), {}))

        sys.modules["diffusers"] = diffusers_stub
        pil_utils_stub.PIL_INTERPOLATION = {
            "linear": 2,
            "bilinear": 2,
            "bicubic": 3,
            "lanczos": 1,
            "nearest": 0,
        }
        utils_stub.pil_utils = pil_utils_stub
        diffusers_stub.utils = utils_stub
        cross_attention_stub.CrossAttention = type("CrossAttention", (), {})
        models_stub.cross_attention = cross_attention_stub
        diffusers_stub.models = models_stub
        sys.modules["diffusers.utils"] = utils_stub
        sys.modules["diffusers.utils.pil_utils"] = pil_utils_stub
        sys.modules["diffusers.models"] = models_stub
        sys.modules["diffusers.models.cross_attention"] = cross_attention_stub


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a LAVIS BLIP-2 retrieval checkpoint to Hugging Face format."
    )
    parser.add_argument(
        "--checkpoint-path",
        required=True,
        help="Path to a LAVIS checkpoint, for example output/.../checkpoint_best.pth.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the Hugging Face model and processor will be written.",
    )
    parser.add_argument(
        "--config-path",
        default=None,
        help=(
            "Optional LAVIS training config. This is only needed when the checkpoint "
            "does not include a saved config."
        ),
    )
    parser.add_argument(
        "--model-type",
        choices=["pretrain", "coco"],
        default=None,
        help=(
            "BLIP-2 retrieval model type to use when no config is available. "
            "Ignored when the checkpoint or --config-path supplies model_type."
        ),
    )
    parser.add_argument(
        "--load-base-weights",
        action="store_true",
        help=(
            "Load the base checkpoint from the LAVIS model config before applying "
            "the trained checkpoint. Use this if your training checkpoint omitted "
            "frozen parameters."
        ),
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help=(
            "Compare LAVIS and Transformers ITM/ITC outputs on the demo image "
            "before saving."
        ),
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device used for optional validation, for example cpu or cuda.",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push the converted model and processor to the Hugging Face Hub.",
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="Target Hub repo id, required when --push-to-hub is set.",
    )
    return parser.parse_args()


def require_transformers_blip2_retrieval():
    try:
        from transformers import (
            BertTokenizer,
            Blip2Config,
            Blip2ForImageTextRetrieval,
            Blip2Processor,
            Blip2QFormerConfig,
            Blip2VisionConfig,
            BlipImageProcessor,
        )
        from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
    except ImportError as exc:
        raise ImportError(
            "This converter requires a Transformers version that provides "
            "Blip2ForImageTextRetrieval. Install a recent Transformers release "
            "or install Transformers from source."
        ) from exc

    return {
        "BertTokenizer": BertTokenizer,
        "Blip2Config": Blip2Config,
        "Blip2ForImageTextRetrieval": Blip2ForImageTextRetrieval,
        "Blip2Processor": Blip2Processor,
        "Blip2QFormerConfig": Blip2QFormerConfig,
        "Blip2VisionConfig": Blip2VisionConfig,
        "BlipImageProcessor": BlipImageProcessor,
        "OPENAI_CLIP_MEAN": OPENAI_CLIP_MEAN,
        "OPENAI_CLIP_STD": OPENAI_CLIP_STD,
    }


def load_checkpoint(checkpoint_path):
    import torch

    parsed_url = urlparse(checkpoint_path)
    if parsed_url.scheme in ("http", "https"):
        checkpoint = torch.hub.load_state_dict_from_url(
            checkpoint_path, map_location="cpu", check_hash=False
        )
    else:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint, checkpoint["model"]
    return checkpoint, checkpoint


def resolve_lavis_model_config(checkpoint, config_path=None, model_type=None):
    from omegaconf import OmegaConf

    install_lavis_optional_dependency_stubs()
    if config_path is not None:
        user_cfg = OmegaConf.load(config_path)
        model_cfg = user_cfg.model
    elif isinstance(checkpoint, dict) and (checkpoint.get("config") or {}).get("model"):
        checkpoint_config = checkpoint.get("config") or {}
        model_cfg = OmegaConf.create(checkpoint_config["model"])
    else:
        model_cfg = OmegaConf.create({"model_type": model_type})

    resolved_model_type = model_cfg.get("model_type", model_type)
    if resolved_model_type is None:
        raise ValueError(
            "Unable to infer the BLIP-2 model_type. Pass --config-path or --model-type."
        )

    from lavis.models.blip2_models.blip2_qformer import Blip2Qformer

    default_cfg = OmegaConf.load(
        Blip2Qformer.default_config_path(model_type=resolved_model_type)
    ).model
    model_cfg = OmegaConf.merge(default_cfg, model_cfg)
    model_cfg.model_type = resolved_model_type

    if model_cfg.get("vit_model", "eva_clip_g") != "eva_clip_g":
        raise ValueError(
            "This converter currently supports the EVA-CLIP-g BLIP-2 retrieval "
            "checkpoints used by the pretrain and coco model types."
        )

    return model_cfg


def build_lavis_retrieval_model(model_cfg, state_dict, load_base_weights=False):
    install_lavis_optional_dependency_stubs()

    from lavis.models.blip2_models.blip2_image_text_matching import Blip2ITM

    model = Blip2ITM(
        vit_model=model_cfg.get("vit_model", "eva_clip_g"),
        img_size=model_cfg.get("image_size"),
        drop_path_rate=model_cfg.get("drop_path_rate", 0),
        use_grad_checkpoint=model_cfg.get("use_grad_checkpoint", False),
        vit_precision=model_cfg.get("vit_precision", "fp16"),
        freeze_vit=model_cfg.get("freeze_vit", True),
        num_query_token=model_cfg.get("num_query_token", 32),
        cross_attention_freq=model_cfg.get("cross_attention_freq", 2),
        embed_dim=model_cfg.get("embed_dim", 256),
        max_txt_len=model_cfg.get("max_txt_len", 32),
    )

    if load_base_weights:
        model.load_checkpoint_from_config(model_cfg)

    load_msg = model.load_state_dict(state_dict, strict=False)
    if load_msg.missing_keys and not load_base_weights:
        missing = ", ".join(load_msg.missing_keys[:10])
        raise RuntimeError(
            "The checkpoint does not contain all model weights. Re-run with "
            "--load-base-weights if the missing parameters were frozen during "
            f"training. First missing keys: {missing}"
        )

    return model.eval()


def build_hf_model_and_processor(model_cfg, transformers_modules):
    BertTokenizer = transformers_modules["BertTokenizer"]
    Blip2Config = transformers_modules["Blip2Config"]
    Blip2ForImageTextRetrieval = transformers_modules["Blip2ForImageTextRetrieval"]
    Blip2Processor = transformers_modules["Blip2Processor"]
    Blip2QFormerConfig = transformers_modules["Blip2QFormerConfig"]
    Blip2VisionConfig = transformers_modules["Blip2VisionConfig"]
    BlipImageProcessor = transformers_modules["BlipImageProcessor"]
    OPENAI_CLIP_MEAN = transformers_modules["OPENAI_CLIP_MEAN"]
    OPENAI_CLIP_STD = transformers_modules["OPENAI_CLIP_STD"]

    image_size = model_cfg.get("image_size")
    tokenizer = BertTokenizer.from_pretrained(
        "bert-base-uncased", truncation_side="right"
    )
    tokenizer.add_special_tokens({"bos_token": "[DEC]"})

    vision_config = Blip2VisionConfig(image_size=image_size).to_dict()
    qformer_config = Blip2QFormerConfig(
        vocab_size=len(tokenizer),
        cross_attention_frequency=model_cfg.get("cross_attention_freq", 2),
        use_qformer_text_input=True,
    ).to_dict()
    config = Blip2Config(
        vision_config=vision_config,
        qformer_config=qformer_config,
        num_query_tokens=model_cfg.get("num_query_token", 32),
        image_text_hidden_size=model_cfg.get("embed_dim", 256),
    )

    model = Blip2ForImageTextRetrieval(config).eval()
    image_processor = BlipImageProcessor(
        size={"height": image_size, "width": image_size},
        image_mean=OPENAI_CLIP_MEAN,
        image_std=OPENAI_CLIP_STD,
    )
    processor = Blip2Processor(image_processor=image_processor, tokenizer=tokenizer)

    return model, processor


def add_vision_key_mappings(rename_keys, num_hidden_layers):
    rename_keys.extend(
        [
            ("visual_encoder.cls_token", "vision_model.embeddings.class_embedding"),
            ("visual_encoder.pos_embed", "vision_model.embeddings.position_embedding"),
            (
                "visual_encoder.patch_embed.proj.weight",
                "vision_model.embeddings.patch_embedding.weight",
            ),
            (
                "visual_encoder.patch_embed.proj.bias",
                "vision_model.embeddings.patch_embedding.bias",
            ),
            ("ln_vision.weight", "vision_model.post_layernorm.weight"),
            ("ln_vision.bias", "vision_model.post_layernorm.bias"),
        ]
    )

    for i in range(num_hidden_layers):
        rename_keys.extend(
            [
                (
                    f"visual_encoder.blocks.{i}.norm1.weight",
                    f"vision_model.encoder.layers.{i}.layer_norm1.weight",
                ),
                (
                    f"visual_encoder.blocks.{i}.norm1.bias",
                    f"vision_model.encoder.layers.{i}.layer_norm1.bias",
                ),
                (
                    f"visual_encoder.blocks.{i}.norm2.weight",
                    f"vision_model.encoder.layers.{i}.layer_norm2.weight",
                ),
                (
                    f"visual_encoder.blocks.{i}.norm2.bias",
                    f"vision_model.encoder.layers.{i}.layer_norm2.bias",
                ),
                (
                    f"visual_encoder.blocks.{i}.attn.qkv.weight",
                    f"vision_model.encoder.layers.{i}.self_attn.qkv.weight",
                ),
                (
                    f"visual_encoder.blocks.{i}.attn.proj.weight",
                    f"vision_model.encoder.layers.{i}.self_attn.projection.weight",
                ),
                (
                    f"visual_encoder.blocks.{i}.attn.proj.bias",
                    f"vision_model.encoder.layers.{i}.self_attn.projection.bias",
                ),
                (
                    f"visual_encoder.blocks.{i}.mlp.fc1.weight",
                    f"vision_model.encoder.layers.{i}.mlp.fc1.weight",
                ),
                (
                    f"visual_encoder.blocks.{i}.mlp.fc1.bias",
                    f"vision_model.encoder.layers.{i}.mlp.fc1.bias",
                ),
                (
                    f"visual_encoder.blocks.{i}.mlp.fc2.weight",
                    f"vision_model.encoder.layers.{i}.mlp.fc2.weight",
                ),
                (
                    f"visual_encoder.blocks.{i}.mlp.fc2.bias",
                    f"vision_model.encoder.layers.{i}.mlp.fc2.bias",
                ),
            ]
        )


def create_rename_keys(num_hidden_layers):
    rename_keys = []
    add_vision_key_mappings(rename_keys, num_hidden_layers)
    rename_keys.extend(
        [
            ("Qformer.bert.embeddings.LayerNorm.weight", "qformer.layernorm.weight"),
            ("Qformer.bert.embeddings.LayerNorm.bias", "qformer.layernorm.bias"),
            (
                "Qformer.bert.embeddings.word_embeddings.weight",
                "embeddings.word_embeddings.weight",
            ),
            (
                "Qformer.bert.embeddings.position_embeddings.weight",
                "embeddings.position_embeddings.weight",
            ),
            ("vision_proj.weight", "vision_projection.weight"),
            ("vision_proj.bias", "vision_projection.bias"),
            ("text_proj.weight", "text_projection.weight"),
            ("text_proj.bias", "text_projection.bias"),
        ]
    )
    return rename_keys


def convert_state_dict_keys(state_dict, num_hidden_layers):
    import torch

    state_dict = dict(state_dict)

    for src, dest in create_rename_keys(num_hidden_layers):
        if src in state_dict:
            state_dict[dest] = state_dict.pop(src)

    for i in range(num_hidden_layers):
        q_bias_key = f"visual_encoder.blocks.{i}.attn.q_bias"
        v_bias_key = f"visual_encoder.blocks.{i}.attn.v_bias"
        if q_bias_key not in state_dict or v_bias_key not in state_dict:
            continue

        q_bias = state_dict.pop(q_bias_key)
        v_bias = state_dict.pop(v_bias_key)
        qkv_bias = torch.cat((q_bias, torch.zeros_like(v_bias), v_bias))
        state_dict[f"vision_model.encoder.layers.{i}.self_attn.qkv.bias"] = qkv_bias

    converted = {}
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith("Qformer.bert"):
            new_key = new_key.replace("Qformer.bert", "qformer", 1)
        if "attention.self" in new_key:
            new_key = new_key.replace("attention.self", "attention.attention")
        converted[new_key] = value

    return converted


def load_hf_state_dict(hf_model, state_dict, num_hidden_layers):
    converted_state_dict = convert_state_dict_keys(state_dict, num_hidden_layers)
    missing_keys, unexpected_keys = hf_model.load_state_dict(
        converted_state_dict, strict=False
    )

    allowed_missing = {"qformer.embeddings.position_ids"}
    missing_keys = [key for key in missing_keys if key not in allowed_missing]
    unexpected_keys = [
        key
        for key in unexpected_keys
        if key != "temp"
        and key != "qformer.embeddings.position_ids"
        and not key.startswith("Qformer.cls")
        and not key.startswith("qformer.cls")
    ]

    if missing_keys:
        raise RuntimeError(f"Missing keys when loading Transformers model: {missing_keys}")
    if unexpected_keys:
        raise RuntimeError(
            f"Unexpected keys when loading Transformers model: {unexpected_keys}"
        )


def validate_conversion(lavis_model, hf_model, processor, image_size, device):
    import torch
    from PIL import Image

    install_lavis_optional_dependency_stubs()
    from lavis.processors.blip_processors import BlipImageEvalProcessor

    with torch.no_grad():
        repo_root = Path(__file__).resolve().parents[2]
        image_path = repo_root / "docs" / "_static" / "merlion.png"
        raw_image = Image.open(image_path).convert("RGB")
        caption = "a large fountain spewing water into the air"

        lavis_processor = BlipImageEvalProcessor(image_size=image_size)
        lavis_pixel_values = lavis_processor(raw_image).unsqueeze(0).to(device)
        hf_inputs = processor(images=raw_image, text=[caption], return_tensors="pt").to(
            device
        )

        lavis_model.to(device)
        hf_model.to(device)

        lavis_itm = lavis_model(
            {"image": lavis_pixel_values, "text_input": [caption]}, match_head="itm"
        )
        hf_itm = hf_model(
            **hf_inputs, use_image_text_matching_head=True
        ).logits_per_image
        if not torch.allclose(lavis_itm, hf_itm, atol=1e-4):
            raise AssertionError("ITM validation failed: LAVIS and Transformers differ.")

        lavis_itc = lavis_model(
            {"image": lavis_pixel_values, "text_input": [caption]}, match_head="itc"
        )
        hf_itc = hf_model(
            **hf_inputs, use_image_text_matching_head=False
        ).logits_per_image
        if not torch.allclose(lavis_itc, hf_itc, atol=1e-4):
            raise AssertionError("ITC validation failed: LAVIS and Transformers differ.")


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    if args.push_to_hub and args.repo_id is None:
        raise ValueError("--repo-id is required when --push-to-hub is set.")

    transformers_modules = require_transformers_blip2_retrieval()
    checkpoint, state_dict = load_checkpoint(args.checkpoint_path)
    model_cfg = resolve_lavis_model_config(
        checkpoint, config_path=args.config_path, model_type=args.model_type
    )

    lavis_model = build_lavis_retrieval_model(
        model_cfg,
        state_dict=state_dict,
        load_base_weights=args.load_base_weights,
    )
    hf_model, processor = build_hf_model_and_processor(model_cfg, transformers_modules)
    load_hf_state_dict(
        hf_model,
        lavis_model.state_dict(),
        num_hidden_layers=hf_model.config.vision_config.num_hidden_layers,
    )

    if args.validate:
        validate_conversion(
            lavis_model,
            hf_model,
            processor,
            image_size=model_cfg.get("image_size"),
            device=args.device,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    processor.save_pretrained(output_dir)
    hf_model.save_pretrained(output_dir)
    LOGGER.info("Saved converted model and processor to %s", output_dir)

    if args.push_to_hub:
        processor.push_to_hub(args.repo_id)
        hf_model.push_to_hub(args.repo_id)
        LOGGER.info("Pushed converted model and processor to %s", args.repo_id)


if __name__ == "__main__":
    main()
