from lavis.common.registry import registry
from lavis.processors.base_processor import BaseProcessor
from lavis.processors.randaugment import RandomAugment
from omegaconf import OmegaConf
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode


class BlipV2ImageBaseProcessor(BaseProcessor):
    def __init__(self, mean=None, std=None):
        if mean is None:
            mean = IMAGENET_DEFAULT_MEAN
        if std is None:
            std = IMAGENET_DEFAULT_STD

        self.normalize = transforms.Normalize(mean, std)


# @registry.register_processor("blip_caption")
# class BlipCaptionProcessor(BaseProcessor):
#     def __init__(self, prompt="", max_words=30):
#         self.prompt = prompt
#         self.max_words = max_words

#     def __call__(self, caption):
#         caption = self.prompt + self.pre_caption(caption)

#         return caption

#     @classmethod
#     def from_config(cls, cfg=None):
#         if cfg is None:
#             cfg = OmegaConf.create()

#         prompt = cfg.get("prompt", "")
#         max_words = cfg.get("max_words", 30)

#         return cls(prompt=prompt, max_words=max_words)

#     def pre_caption(self, caption):
#         caption = re.sub(
#             r"([.!\"()*#:;~])",
#             " ",
#             caption.lower(),
#         )
#         caption = re.sub(
#             r"\s{2,}",
#             " ",
#             caption,
#         )
#         caption = caption.rstrip("\n")
#         caption = caption.strip(" ")

#         # truncate caption
#         caption_words = caption.split(" ")
#         if len(caption_words) > self.max_words:
#             caption = " ".join(caption_words[: self.max_words])

#         return caption


@registry.register_processor("blipv2_question")
class BlipV2QuestionProcessor(BaseProcessor):
    def __init__(self, max_words=100):
        self.max_words = max_words

    def __call__(self, question):
        return self.pre_question(question)

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        max_words = cfg.get("max_words", 100)

        return cls(max_words=max_words)

    def pre_question(self, question):
        # question = re.sub(
        #     r"([.!\"()*#:;~])",
        #     "",
        #     question.lower(),
        # )
        # question = question.rstrip(" ")

        # truncate question
        question_words = question.split(" ")
        if len(question_words) > self.max_words:
            question = " ".join(question_words[: self.max_words])

        return question


@registry.register_processor("blipv2_image_train")
class BlipV2ImageTrainProcessor(BlipV2ImageBaseProcessor):
    def __init__(
        self, image_size=224, mean=None, std=None, min_scale=0.5, max_scale=1.0
    ):
        super().__init__(mean=mean, std=std)

        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(min_scale, max_scale),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(),
                RandomAugment(
                    2,
                    5,
                    isPIL=True,
                    augs=[
                        "Identity",
                        "AutoContrast",
                        "Brightness",
                        "Sharpness",
                        "Equalize",
                        "ShearX",
                        "ShearY",
                        "TranslateX",
                        "TranslateY",
                        "Rotate",
                    ],
                ),
                transforms.ToTensor(),
                self.normalize,
            ]
        )

    def __call__(self, item):
        return self.transform(item)

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", 224)

        mean = cfg.get("mean", IMAGENET_DEFAULT_MEAN)
        std = cfg.get("std", IMAGENET_DEFAULT_STD)

        min_scale = cfg.get("min_scale", 0.5)
        max_scale = cfg.get("max_scale", 1.0)

        return cls(
            image_size=image_size,
            mean=mean,
            std=std,
            min_scale=min_scale,
            max_scale=max_scale,
        )


@registry.register_processor("blipv2_image_eval")
class BlipV2ImageEvalProcessor(BlipV2ImageBaseProcessor):
    def __init__(self, image_size=224, mean=None, std=None):
        super().__init__(mean=mean, std=std)

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                self.normalize,
            ]
        )

    def __call__(self, item):
        return self.transform(item)

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", 224)

        mean = cfg.get("mean", IMAGENET_DEFAULT_MEAN)
        std = cfg.get("std", IMAGENET_DEFAULT_STD)

        return cls(image_size=image_size, mean=mean, std=std)
