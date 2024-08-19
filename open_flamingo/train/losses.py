from open_flamingo.src.vlm import VLM
import torch
from typing import List, Optional

SUPPORTED_LOSSES = ["next_token_prediction",
                    "supervised_finetune"]


def get_loss_fn(loss_name):
    if loss_name == "next_token_prediction":
        return NextTokenPrediction()
    elif loss_name == "supervised_finetune":
        return SupervisedPrediction()
    else:
        raise ValueError(
            f"Loss {loss_name} not supported. Supported losses: {SUPPORTED_LOSSES}"
        )

class Loss:
    @property
    def name(self):
        raise NotImplementedError

    def __call__(
        self,
        model: VLM,
        tokenizer,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        autocast: callable,
    ):
        """
        Args:
            model: VLM model
            images: images tensor, already moved to device and cast to appropriate dtype
                shape (B, T_img, F, C, H, W)
            input_ids: input ids tensor, already moved to device and cast to appropriate dtype
                shape (B, T_text)
            attention_mask: attention mask tensor, already moved to device and cast to appropriate dtype
                shape (B, T_text)
            autocast: autocast context manager
        Return:
            loss: scalar loss
        """
        raise NotImplementedError


class NextTokenPrediction(Loss):
    @property
    def name(self):
        return "next_token_prediction"

    def __call__(
        self,
        model: VLM,
        tokenizer,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        autocast: callable,
    ):
        # set up labels; language model is expected to handle shifting
        labels = input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100
        special_token_ids = torch.Tensor(unwrap_model(model).special_token_ids).to(
            labels.device
        )
        labels[torch.isin(labels, special_token_ids)] = -100 # TODO: dont want to remove loss on <|endofchunk|> tokens

        labels = labels.to(input_ids.device)

        # call forward
        with autocast():
            loss = model(
                vision_x=images,
                lang_x=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )[0]
        return loss


class SupervisedPrediction(Loss):
    @property
    def name(self):
        return "supervised_finetune"

    def __call__(
        self,
        model: VLM,
        tokenizer,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
        autocast: callable,
        image_size: Optional[torch.Tensor] = None,
    ):
        # set up labels; language model is expected to handle shifting
        labels[labels == tokenizer.pad_token_id] = -100
        special_token_ids = torch.Tensor(unwrap_model(model).special_token_ids).to(
            labels.device
        )
        labels[torch.isin(labels, special_token_ids)] = -100 # TODO: dont want to remove loss on <|endofchunk|> tokens

        labels = labels.to(input_ids.device)

        # call forward
        with autocast():
            loss = model(
                vision_x=images,
                image_size=image_size,
                lang_x=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )[0]
        return loss


def unwrap_model(model):
    """
    Unwrap a model from a DataParallel or DistributedDataParallel wrapper.
    """
    if isinstance(
        model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)
    ):
        return model.module
    else:
        return model
