from copy import deepcopy

import torch
import torch.nn.functional as F

from torch import nn
from transformers import BertConfig

from lavis.common.registry import registry
from lavis.common.utils import get_abs_path
from lavis.models.med import BertModel
from lavis.models.vit import VisionTransformerEncoder
from lavis.models.base_model import BaseModel, MomentumDistilationMixin
from lavis.models.albef_models import init_tokenizer, load_from_pretrained


@registry.register_model("albef_nlvr")
class AlbefNLVR(BaseModel, MomentumDistilationMixin):
    PRETRAINED_MODEL_DICT = {
        "base": "configs/models/albef_nlvr_base.yaml",
    }

    def __init__(
        self,
        image_encoder,
        text_encoder,
        num_classes,
        momentum=0.995,
        alpha=0.4,
        use_distill=True,
        max_txt_len=40,
    ):
        super().__init__()

        self.tokenizer = init_tokenizer()
        self.max_txt_len = max_txt_len

        self.use_distill = use_distill

        self.visual_encoder = image_encoder
        self.text_encoder = text_encoder

        hidden_size = text_encoder.config.hidden_size
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )

        self.share_cross_attention(self.text_encoder.encoder)

        if self.use_distill:
            self.visual_encoder_m = deepcopy(self.visual_encoder)
            self.text_encoder_m = deepcopy(self.text_encoder)
            self.cls_head_m = deepcopy(self.cls_head)

            self.share_cross_attention(self.text_encoder_m.encoder)

            self.momentum = momentum
            self.alpha = alpha

            self.model_pairs = [
                [self.visual_encoder, self.visual_encoder_m],
                [self.text_encoder, self.text_encoder_m],
                [self.cls_head, self.cls_head_m],
            ]

            self.copy_params()

    def _rampup_factor(self, epoch, iters, num_iters_per_epoch):
        return min(1, (epoch * num_iters_per_epoch + iters) / (2 * num_iters_per_epoch))

    def forward(self, samples, is_train=True):
        text = samples["text_input"]
        text = self.tokenizer(
            text,
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(self.device)

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
            if self.use_distill:
                with torch.no_grad():
                    self._momentum_update()

                    image_embeds_m = self.visual_encoder_m(images)
                    image0_embeds_m, image1_embeds_m = torch.split(
                        image_embeds_m, targets.size(0)
                    )
                    multimodal_embeds_m = self.text_encoder(
                        text.input_ids,
                        attention_mask=text.attention_mask,
                        encoder_hidden_states=[image0_embeds_m, image1_embeds_m],
                        encoder_attention_mask=[
                            image_atts[: image0_embeds_m.size(0)],
                            image_atts[image0_embeds_m.size(0) :],
                        ],
                        return_dict=True,
                    )

                    prediction_m = self.cls_head_m(
                        multimodal_embeds_m.last_hidden_state[:, 0, :]
                    )

                alpha = self.alpha * self._rampup_factor(
                    epoch=samples["epoch"],
                    iters=samples["iters"],
                    num_iters_per_epoch=samples["num_iters_per_epoch"],
                )

                loss = (1 - alpha) * F.cross_entropy(
                    prediction, targets
                ) - alpha * torch.sum(
                    F.log_softmax(prediction, dim=1) * F.softmax(prediction_m, dim=1),
                    dim=1,
                ).mean()
            else:
                loss = F.cross_entropy(prediction, targets)

            return {"loss": loss}
        else:
            return {"predictions": prediction, "targets": targets}

    def share_cross_attention(self, model):
        for i in range(6):
            layer_num = 6 + i * 2
            modules_0 = model.layer[layer_num].crossattention.self._modules
            modules_1 = model.layer[layer_num + 1].crossattention.self._modules

            for name in modules_0.keys():
                if "key" in name or "value" in name:
                    module_0 = modules_0[name]
                    module_1 = modules_1[name]
                    if hasattr(module_0, "weight"):
                        module_0.weight = module_1.weight
                        if hasattr(module_0, "bias"):
                            module_0.bias = module_1.bias

    def predict(self, samples):
        output = self.forward(samples, is_train=False)
        return output

    @classmethod
    def from_config(cls, cfg=None):
        image_encoder = VisionTransformerEncoder.from_config(cfg)

        # text encoder + multimodal encoder
        bert_config = BertConfig.from_json_file(get_abs_path(cfg["med_config_path"]))
        bert_config.num_hidden_layers = 18

        text_encoder = BertModel.from_pretrained(
            "bert-base-uncased", config=bert_config, add_pooling_layer=False
        )

        alpha = cfg.get("alpha", 0.4)
        momentum = cfg.get("momentum", 0.995)
        use_distill = cfg.get("use_distill", True)
        num_classes = cfg.get("num_classes", -1)
        max_txt_len = cfg.get("max_txt_len", 40)

        assert num_classes > 1, "Invalid number of classes provided, found {}".format(
            num_classes
        )

        model = cls(
            image_encoder=image_encoder,
            text_encoder=text_encoder,
            use_distill=use_distill,
            alpha=alpha,
            num_classes=num_classes,
            momentum=momentum,
            max_txt_len=max_txt_len,
        )

        # load pre-trained weights
        pretrain_path = cfg.get("pretrained", None)
        if pretrain_path is not None:
            model, msg = load_from_pretrained(model, url_or_filename=pretrain_path)

        return model
