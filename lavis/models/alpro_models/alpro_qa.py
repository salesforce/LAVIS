from lib2to3.pgen2 import token
import torch
import torch.nn.functional as F
from torch import nn
from lavis.common.registry import registry
from lavis.common.config import node_to_dict
from lavis.models.alpro_models import init_tokenizer, load_from_pretrained
from lavis.models.base_model import BaseModel
from lavis.models.med import XBertEncoder
from lavis.models.timesformer.vit import TimeSformer


@registry.register_model("alpro_qa")
class AlproQA(BaseModel):
    def __init__(
        self, visual_encoder, text_encoder, hidden_size, num_classes, max_txt_len=40
    ):
        super().__init__()

        self.tokenizer = init_tokenizer()

        self.visual_encoder = visual_encoder

        self.text_encoder = text_encoder

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(True),
            nn.Linear(hidden_size * 2, num_classes),
        )

        self.max_txt_len = max_txt_len

    @classmethod
    def default_config_path(cls, model_type="base"):
        paths = {
            "base": "lavis/configs/models/alpro_qa_base.yaml",
        }

        assert model_type in paths, "Unknown model type {}".format(model_type)
        return paths[model_type]

    def forward(self, samples, is_train=False):
        visual_inputs = samples["video"]
        question = samples["text_input"]
        targets = samples["answers"]

        # forward text
        text = self.tokenizer(
            question,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(self.device)

        text_output = self.text_encoder.forward_text_embeds(
            text,
            token_type_ids=torch.zeros(
                text.input_ids.shape, dtype=torch.long, device=self.device
            ),
        )
        text_embeds = text_output.last_hidden_state

        # forward visual
        # timeSformer asks for (b, c, t, h, w) as input.
        image_embeds = self.visual_encoder.forward_features(visual_inputs)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            self.device
        )

        # forward cross-encoder
        attention_mask = torch.cat([text.attention_mask, image_atts], dim=1)
        embedding_output = torch.cat([text_embeds, image_embeds], dim=1)

        output = super(type(self.text_encoder), self.text_encoder).forward(
            encoder_embeds=embedding_output,
            attention_mask=attention_mask,
            return_dict=True,
            mode="fusion",
        )

        prediction = self.classifier(output.last_hidden_state[:, 0, :])
        if is_train:
            loss = F.cross_entropy(prediction, targets)
            return {"loss": loss}
        else:
            return {"predictions": prediction, "targets": targets}

    def predict(self, samples):
        output = self.forward(samples, is_train=False)
        return output

    @classmethod
    def _build_from_cfg(cls, cfg):
        # vision encoder
        visual_encoder_config = node_to_dict(cfg.timesformer)
        visual_encoder = TimeSformer(**visual_encoder_config)

        # text encoder
        text_encoder = XBertEncoder.build_from_cfg(cfg)

        num_classes = cfg.get("num_classes", -1)
        hidden_size = cfg.get("hidden_size", 768)

        model = cls(
            visual_encoder=visual_encoder,
            text_encoder=text_encoder,
            hidden_size=hidden_size,
            num_classes=num_classes,
        )

        pretrain_path = cfg.get("pretrained")

        if pretrain_path is not None:
            model, msg = load_from_pretrained(model, url_or_filename=pretrain_path)

        return model
