from lavis.common.registry import registry
from lavis.tasks.base_task import BaseTask
from lavis.tasks.captioning import CaptionTask
from lavis.tasks.image_text_pretrain import ImageTextPretrainTask
from lavis.tasks.multimodal_classification import (
    MultimodalClassificationTask,
    VideoQATask,
    VisualEntailmentTask,
)
from lavis.tasks.retrieval import RetrievalTask
from lavis.tasks.vqa import VQATask


def setup_task(cfg):
    assert "task" in cfg.run_cfg, "Task name must be provided."

    task_name = cfg.run_cfg.task
    task = registry.get_task_class(task_name).setup_task(cfg=cfg)
    assert task is not None, "Task {} not properly registered.".format(task_name)

    return task


__all__ = [
    "BaseTask",
    "RetrievalTask",
    "CaptionTask",
    "VQATask",
    "MultimodalClassificationTask",
    "VideoQATask",
    "VisualEntailmentTask",
    "ImageTextPretrainTask",
]
