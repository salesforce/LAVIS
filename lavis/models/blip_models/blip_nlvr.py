import os

import torch
import torch.nn.functional as F
from lavis.common.registry import registry
from lavis.common.utils import get_abs_path, is_url
from lavis.models.base_model import MomentumDistilationMixin
from lavis.models.blip_models.blip import BlipBase
from lavis.models.blip_models.nlvr_encoder import BertModel
from lavis.models.vit import VisionTransformerEncoder, interpolate_pos_embed
from timm.models.hub import download_cached_file
from torch import nn
from transformers import BertConfig


@registry.register_model("blip_nlvr")
class BlipNLVR(BlipBase, MomentumDistilationMixin):
    PRETRAINED_MODEL_DICT = {
        "base": "configs/models/blip_nlvr_base.yaml",
    }

    def __init__(
        self,
        image_encoder,
        text_encoder,
        num_classes,
        max_txt_len=40,
        use_distill=True,
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()

        self.use_distill = use_distill

        self.visual_encoder = image_encoder
        self.text_encoder = text_encoder

        hidden_size = text_encoder.config.hidden_size
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )

        self.max_txt_len = max_txt_len

    def _rampup_factor(self, epoch, iters, num_iters_per_epoch):
        return min(1, (epoch * num_iters_per_epoch + iters) / num_iters_per_epoch)

    def forward(self, samples, is_train=True):
        text = samples["text_input"]
        text = self.tokenizer(
            text,
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(self.device)
        text.input_ids[:, 0] = self.tokenizer.enc_token_id

        targets = samples["label"]

        image0 = samples["image0"]
        image1 = samples["image1"]
        images = torch.cat([image0, image1], dim=0)

        image_embeds = self.visual_encoder.forward_features(images)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            self.device
        )
        image0_embeds, image1_embeds = torch.split(image_embeds, targets.size(0))

        multimodal_embeds = self.text_encoder(
            text.input_ids,
            attention_mask=text.attention_mask,
            encoder_hidden_states=[image0_embeds, image1_embeds],
            encoder_attention_mask=[
                image_atts[: image0_embeds.size(0)],
                image_atts[image0_embeds.size(0) :],
            ],
            return_dict=True,
        )

        prediction = self.cls_head(multimodal_embeds.last_hidden_state[:, 0, :])

        if is_train:
            loss = F.cross_entropy(prediction, targets)
            return {"loss": loss}
        else:
            return {"predictions": prediction, "targets": targets}

    def predict(self, samples):
        output = self.forward(samples, is_train=False)
        return output

    @classmethod
    def from_config(cls, cfg=None):
        image_encoder = VisionTransformerEncoder.from_config(cfg)

        # text encoder + multimodal encoder
        bert_config = BertConfig.from_json_file(get_abs_path(cfg["med_config_path"]))
        text_encoder = BertModel(config=bert_config, add_pooling_layer=False)

        num_classes = cfg.get("num_classes", -1)
        max_txt_len = cfg.get("max_txt_len", 40)

        assert num_classes > 1, "Invalid number of classes provided, found {}".format(
            num_classes
        )

        model = cls(
            image_encoder=image_encoder,
            text_encoder=text_encoder,
            num_classes=num_classes,
            max_txt_len=max_txt_len,
        )

        # load pre-trained weights
        pretrain_path = cfg.get("pretrained", None)
        if pretrain_path is not None:
            msg = model.load_from_pretrained(url_or_filename=pretrain_path)

        return model

    def load_from_pretrained(self, url_or_filename):
        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location="cpu")
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location="cpu")
        else:
            raise RuntimeError("checkpoint url or path is invalid")
        state_dict = checkpoint["model"]

        state_dict["visual_encoder.pos_embed"] = interpolate_pos_embed(
            state_dict["visual_encoder.pos_embed"], self.visual_encoder
        )

        for key in list(state_dict.keys()):
            if "crossattention.self." in key:
                new_key0 = key.replace("self", "self0")
                new_key1 = key.replace("self", "self1")
                state_dict[new_key0] = state_dict[key]
                state_dict[new_key1] = state_dict[key]
            elif "crossattention.output.dense." in key:
                new_key0 = key.replace("dense", "dense0")
                new_key1 = key.replace("dense", "dense1")
                state_dict[new_key0] = state_dict[key]
                state_dict[new_key1] = state_dict[key]

        msg = self.load_state_dict(state_dict, strict=False)
        print("load checkpoint from %s" % url_or_filename)
        return msg
