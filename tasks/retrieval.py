from tasks.base_task import BaseTask

from common.registry import registry


@registry.register_task('retrieval')
class RetrievalTask(BaseTask):
    def __init__(self):
        super().__init__()
    
    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg

        return cls()
