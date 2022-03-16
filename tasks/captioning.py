from tasks.base_task import BaseTask

from common.registry import registry


@registry.register_task('captioning')
class CaptionTask(BaseTask):
    def __init__(self, cfg):
        super().__init__(cfg)