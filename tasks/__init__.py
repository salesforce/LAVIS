from common.registry import registry

from tasks.base_task import BaseTask
from tasks.retrieval import RetrievalTask

def setup_task(cfg):
    assert 'task' in cfg.get_runner_config(), 'Task name must be provided.'

    task_name = cfg.get_runner_config().task
    task = registry.get_task_class(task_name)(cfg)
    assert task is not None, "Task {} not properly registered.".format(task_name)

    return task

__all__ = [
        'BaseTask', 'RetrievalTask',
        ]