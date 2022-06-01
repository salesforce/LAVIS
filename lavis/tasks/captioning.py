import os
import json

from lavis.tasks.base_task import BaseTask
from lavis.common.registry import registry

import lavis.common.utils as utils


@registry.register_task("captioning")
class CaptionTask(BaseTask):
    ID_KEY = "image_id"
    CAP_KEY = "caption"

    def __init__(self, num_beams, max_len, min_len, evaluate):
        super().__init__()

        self.num_beams = num_beams
        self.max_len = max_len
        self.min_len = min_len
        self.evaluate = evaluate

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg

        num_beams = run_cfg.num_beams
        max_len = run_cfg.max_len
        min_len = run_cfg.min_len
        evaluate = run_cfg.evaluate

        return cls(
            num_beams=num_beams, max_len=max_len, min_len=min_len, evaluate=evaluate
        )

    def valid_step(self, model, samples):
        results = []

        # run_cfg = slf.cfg.run_cfg
        captions = model.generate(
            samples,
            use_nucleus_sampling=False,
            num_beams=self.num_beams,
            max_length=self.max_len,
            min_length=self.min_len,
        )

        indices = samples[self.ID_KEY]
        for caption, index in zip(captions, indices):
            # results.append({"image_id": img_id.item(), "caption": caption})
            results.append({self.ID_KEY: index.item(), self.CAP_KEY: caption})

        return results

    def after_evaluation(self, val_result, split_name, epoch, **kwargs):
        eval_result_file = utils.save_result(
            result=val_result,
            result_dir=registry.get_path("result_dir"),
            filename="{}_epoch{}".format(split_name, epoch),
            remove_duplicate=self.ID_KEY,
        )

        metrics = self._report_metrics(
            eval_result_file=eval_result_file, split_name=split_name
        )

        return metrics

    @utils.main_process
    def _report_metrics(self, eval_result_file, split_name):

        # TODO better way to define this
        coco_gt_root = "annotation/coco_gt"

        # coco_val = utils.coco_caption_eval(self.config['coco_gt_root'], val_result_file, 'val')
        coco_val = utils.coco_caption_eval(coco_gt_root, eval_result_file, split_name)

        agg_metrics = coco_val.eval["CIDEr"] + coco_val.eval["Bleu_4"]
        log_stats = {split_name: {k: v for k, v in coco_val.eval.items()}}

        with open(
            os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
        ) as f:
            f.write(json.dumps(log_stats) + "\n")

        coco_res = {k: v for k, v in coco_val.eval.items()}
        coco_res["agg_metrics"] = agg_metrics

        return coco_res
