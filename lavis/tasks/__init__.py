"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.common.registry import registry
from lavis.tasks.base_task import BaseTask
from lavis.tasks.captioning import CaptionTask
from lavis.tasks.image_text_pretrain import ImageTextPretrainTask
from lavis.tasks.multimodal_classification import (
    MultimodalClassificationTask,
)
from lavis.tasks.retrieval import RetrievalTask
from lavis.tasks.vqa import VQATask, GQATask, AOKVQATask, DisCRNTask
from lavis.tasks.vqa_reading_comprehension import VQARCTask, GQARCTask
from lavis.tasks.dialogue import DialogueTask
from lavis.tasks.text_to_image_generation import TextToImageGenerationTask


def setup_task(cfg):
    assert "task" in cfg.run_cfg, "Task name must be provided."

    task_name = cfg.run_cfg.task
    task = registry.get_task_class(task_name).setup_task(cfg=cfg)
    assert task is not None, "Task {} not properly registered.".format(task_name)

    return task


__all__ = [
    "BaseTask",
    "AOKVQATask",
    "RetrievalTask",
    "CaptionTask",
    "VQATask",
    "GQATask",
    "VQARCTask",
    "GQARCTask",
    "MultimodalClassificationTask",
    # "VideoQATask",
    # "VisualEntailmentTask",
    "ImageTextPretrainTask",
    "DialogueTask",
    "TextToImageGenerationTask",
    "DisCRNTask"
]
