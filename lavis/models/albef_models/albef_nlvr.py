"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from copy import deepcopy

import torch
import torch.nn.functional as F
from lavis.common.registry import registry
from lavis.common.utils import get_abs_path
from lavis.models.albef_models import AlbefBase
from lavis.models.albef_models.albef_outputs import AlbefIntermediateOutput, AlbefOutput
from lavis.models.base_model import MomentumDistilationMixin
from lavis.models.med import BertModel
from lavis.models.vit import VisionTransformerEncoder
from torch import nn
from transformers import BertConfig


@registry.register_model("albef_nlvr")
class AlbefNLVR(AlbefBase, MomentumDistilationMixin):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "nlvr": "configs/models/albef_nlvr.yaml",
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

        self.tokenizer = self.init_tokenizer()
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
        """
        Forward function for training and evaluation.

        Args:
            samples (dict): a dict of input samples, which contains the following keys:
                - image0 (torch.Tensor): input image 0, shape (batch_size, 3, H, W), default H=384, W=384.
                - image1 (torch.Tensor): input image 1, shape (batch_size, 3, H, W), default H=384, W=384.
                - text_input (list): list of strings, each string is a natural language sentence.
                - label (torch.LongTensor): ground truth label with shape (batch_size,).
            is_train (bool): whether the model is in training mode.
                If True, the model will return the loss;
                If False, the model will return the prediction.

        Examples:
            >>> import torch
            >>> from lavis.models import load_model
            >>> model = load_model("albef_nlvr")
            >>> samples = {
            ...     "image0": torch.randn(2, 3, 384, 384),
            ...     "image1": torch.randn(2, 3, 384, 384),
            ...     "text_input": ["there is a ferret in tall grass", "there are lips in one of the images"],
            ...     "label": torch.tensor([0, 1]),
            ... }
            >>> output = model(samples)
            >>> output.keys()
            odict_keys(['intermediate_output', 'loss'])
        """
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

        encoder_output = self.text_encoder(
            text.input_ids,
            attention_mask=text.attention_mask,
            encoder_hidden_states=[image0_embeds, image1_embeds],
            encoder_attention_mask=[
                image_atts[: image0_embeds.size(0)],
                image_atts[image0_embeds.size(0) :],
            ],
            return_dict=True,
        )

        prediction = self.cls_head(encoder_output.last_hidden_state[:, 0, :])

        if is_train:
            if self.use_distill:
                with torch.no_grad():
                    self._momentum_update()

                    image_embeds_m = self.visual_encoder_m(images)
                    image0_embeds_m, image1_embeds_m = torch.split(
                        image_embeds_m, targets.size(0)
                    )
                    encoder_output_m = self.text_encoder(
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
                        encoder_output_m.last_hidden_state[:, 0, :]
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

                encoder_output_m = None
                image0_embeds_m, image1_embeds_m = None, None

            # return {"loss": loss}
            return AlbefOutput(
                loss=loss,
                intermediate_output=AlbefIntermediateOutput(
                    image_embeds=torch.stack([image0_embeds, image1_embeds], dim=0),
                    image_embeds_m=torch.stack(
                        [image0_embeds_m, image1_embeds_m], dim=0
                    ),
                    encoder_output=encoder_output,
                    encoder_output_m=encoder_output_m,
                ),
            )
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

    def load_from_pretrained(self, url_or_filename, use_distill=True):
        _, msg = super().load_from_pretrained(url_or_filename)

        if use_distill and any(["_m" in k for k in msg.missing_keys]):
            # this is required when initializing the model from TA pre-trained weights
            self.copy_params()

        return msg

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

        model.load_checkpoint_from_config(cfg)

        return model
