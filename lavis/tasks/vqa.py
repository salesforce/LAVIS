import json
import logging
import os

import common.utils as utils
import torch.distributed as dist
from common.registry import registry
from common.vqa_tools.vqa import VQA
from common.vqa_tools.vqa_eval import VQAEval

from tasks.base_task import BaseTask


@registry.register_task("vqa")
class VQATask(BaseTask):
    def __init__(
        self,
        num_beams,
        max_len,
        min_len,
        evaluate,
        num_ans_candidates,
        inference_method="rank",
    ):
        super().__init__()

        self.num_beams = num_beams
        self.max_len = max_len
        self.min_len = min_len

        self.evaluate = evaluate
        self.inference_method = inference_method
        self.num_ans_candidates = num_ans_candidates

        self.answer_list = None

        self.ques_files = dict()
        self.anno_files = dict()

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg

        num_beams = run_cfg.get("num_beams", 3)
        max_len = run_cfg.get("max_len", 10)
        min_len = run_cfg.get("min_len", 1)

        evaluate = run_cfg.get("evaluate", False)

        inference_method = run_cfg.get("inference_method", "rank")
        num_ans_candidates = run_cfg.get("num_ans_candidates", 128)

        return cls(
            num_beams=num_beams,
            max_len=max_len,
            min_len=min_len,
            evaluate=evaluate,
            num_ans_candidates=num_ans_candidates,
            inference_method=inference_method,
        )

    def build_datasets(self, cfg):
        datasets = super().build_datasets(cfg)

        for split in datasets:
            if (
                hasattr(datasets[split], "coco_fmt_qust_file")
                and datasets[split].coco_fmt_qust_file is not None
            ):
                self.ques_files[split] = datasets[split].coco_fmt_qust_file
                self.anno_files[split] = datasets[split].coco_fmt_anno_file

        assert "test" in datasets, "No testing split is present."
        self.answer_list = datasets["test"].answer_list
        return datasets

    def valid_step(self, model, samples):
        answers = model.predict_answers(
            samples=samples,
            answer_list=self.answer_list,
            inference_method=self.inference_method,
            num_beams=self.num_beams,
            max_len=self.max_len,
            min_len=self.min_len,
            num_ans_candidates=self.num_ans_candidates,
        )

        pred_qa_pairs = []

        question_id = samples["question_id"]
        for answer, ques_id in zip(answers, question_id):
            ques_id = int(ques_id.item())
            pred_qa_pairs.append({"question_id": ques_id, "answer": answer})

        return pred_qa_pairs

    def after_evaluation(self, val_result, result_dir, split_name, **kwargs):
        result_file = save_result(
            val_result,
            result_dir,
            filename="vqa_result",
            remove_duplicate="question_id",
        )

        metrics = self._report_metrics(result_file=result_file, split=split_name)

        return metrics

    @utils.main_process
    def _report_metrics(self, result_file, split):
        metrics = {}

        if split in self.ques_files and split in self.anno_files:
            vqa = VQA(self.anno_files[split], self.ques_files[split])
            vqa_result = vqa.loadRes(
                resFile=result_file, quesFile=self.ques_files[split]
            )

            # create vqaEval object by taking vqa and vqaRes
            # n is precision of accuracy (number of places after decimal), default is 2
            vqa_scorer = VQAEval(vqa, vqa_result, n=2)
            logging.info("Start VQA evaluation.")
            vqa_scorer.evaluate()

            # print accuracies
            overall_acc = vqa_scorer.accuracy["overall"]
            metrics["agg_metrics"] = overall_acc

            logging.info("Overall Accuracy is: %.02f\n" % overall_acc)
            logging.info("Per Answer Type Accuracy is the following:")

            for ans_type in vqa_scorer.accuracy["perAnswerType"]:
                logging.info(
                    "%s : %.02f"
                    % (ans_type, vqa_scorer.accuracy["perAnswerType"][ans_type])
                )
                metrics["ansType"] = vqa_scorer.accuracy["perAnswerType"][ans_type]

        return metrics


def save_result(result, result_dir, filename, remove_duplicate=""):
    result_file = os.path.join(
        result_dir, "%s_rank%d.json" % (filename, utils.get_rank())
    )
    final_result_file = os.path.join(result_dir, "%s.json" % filename)

    json.dump(result, open(result_file, "w"))

    dist.barrier()

    if utils.is_main_process():
        logging.info("Merging results from all ranks.")
        # combine results from all processes
        result = []

        for rank in range(utils.get_world_size()):
            result_file = os.path.join(result_dir, "%s_rank%d.json" % (filename, rank))
            res = json.load(open(result_file, "r"))
            result += res

        if remove_duplicate:
            result_new = []
            id_list = []
            for res in result:
                if res[remove_duplicate] not in id_list:
                    id_list.append(res[remove_duplicate])
                    result_new.append(res)
            result = result_new

        json.dump(result, open(final_result_file, "w"))
        logging.info("result file saved to %s" % final_result_file)

    return final_result_file
