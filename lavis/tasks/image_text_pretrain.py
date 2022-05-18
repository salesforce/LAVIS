from common.registry import registry
from tasks.base_task import BaseTask


@registry.register_task("image_text_pretrain")
class ImageTextPretrainTask(BaseTask):
    def __init__(self):
        super().__init__()
    
    def evaluation(self, model, data_loader, cuda_enabled=True):
        pass