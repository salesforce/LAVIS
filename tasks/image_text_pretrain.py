from common.registry import registry
from tasks.base_task import BaseTask


@registry.register_task("image_text_pretrain")
class ImageTextPretrainTask(BaseTask):
    def __init__(self):
        super().__init__()

    @classmethod
    def setup_task(cls, cfg):
        # run_cfg = cfg.run_cfg

        # num_beams = run_cfg.get("num_beams", 3)
        # max_len = run_cfg.get("max_len", 10)
        # min_len = run_cfg.get("min_len", 1)

        # evaluate = run_cfg.get("evaluate", False)

        # inference_method = run_cfg.get("inference_method", "rank")
        # num_ans_candidates = run_cfg.get("num_ans_candidates", 128)

        return cls(
            # num_beams=num_beams,
            # max_len=max_len,
            # min_len=min_len,
            # evaluate=evaluate,
            # num_ans_candidates=num_ans_candidates,
            # inference_method=inference_method,
        )

    # def build_datasets(self, cfg):
    #     datasets = super().build_datasets(cfg)

    #     return datasets
