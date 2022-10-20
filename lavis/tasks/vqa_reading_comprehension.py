"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
import json
import os
import torch
import torch.distributed as dist
from itertools import chain

import lavis.common.dist_utils as dist_utils
from lavis.common.dist_utils import get_rank, get_world_size, is_main_process
from lavis.common.registry import registry
from lavis.common.vqa_tools.vqa_eval import VQAEval as VQATool
from lavis.tasks.vqa import VQATask


@registry.register_task("vqa_reading_comprehension")
class VQARCTask(VQATask):
    def __init__(
        self,
        num_beams,
        max_len,
        min_len,
        evaluate,
        num_ans_candidates,
        inference_method="rank",
        **kwargs,
    ):
        super().__init__(num_beams, max_len, min_len, evaluate, num_ans_candidates, inference_method)

        self.config = kwargs.get('config')

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
            config=run_cfg,
        )

    def valid_step(self, model, samples):
        answers, captions, gradcams = model.predict_answers(
            samples=samples,
            inference_method=self.inference_method,
            num_beams=self.num_beams,
            max_len=self.max_len,
            min_len=self.min_len,
            internal_bsz_fid=self.config['internal_bsz_fid'],
            num_captions=self.config['num_captions'],
            num_captions_fid=self.config['num_captions_fid'],
            cap_max_length=self.config['cap_max_length'],
            cap_min_length=self.config['cap_min_length'],
            top_k=self.config['top_k'],
            top_p=self.config['top_p'],
            repetition_penalty=self.config['repetition_penalty'],
            num_patches=self.config['num_patches'],
            block_num=self.config['block_num'],
        )

        pred_qa_pairs = []
        sample_captions = []
        sample_gradcams = []

        question_id = samples["question_id"]
        for answer, caption, gradcam, ques_id in zip(answers, captions, gradcams, question_id):
            ques_id = int(ques_id.item())
            pred_qa_pairs.append({"question_id": ques_id, "answer": answer})
            sample_captions.append({"question_id": ques_id, "caption": caption})
            sample_gradcams.append({"question_id": ques_id, "gradcam": gradcam})

        return [sample_gradcams, sample_captions, pred_qa_pairs]

    def after_evaluation(self, val_result, split_name, **kwargs):
        result_ = list(chain(*val_result[0::3]))
        result_file = self.save_gradcam(
            result_,
            result_dir=registry.get_path("result_dir"),
            filename=f"{split_name}_gradcam_result",
            remove_duplicate="question_id",
        )

        result_ = list(chain(*val_result[1::3]))
        result_file = self.save_result(
            result_,
            result_dir=registry.get_path("result_dir"),
            filename=f"{split_name}_caption_result",
            remove_duplicate="question_id",
        )

        result_ = list(chain(*val_result[2::3]))
        result_file = self.save_result(
            result_,
            result_dir=registry.get_path("result_dir"),
            filename=f"{split_name}_vqa_result",
            remove_duplicate="question_id",
        )

        metrics = self._report_metrics(result_file=result_file, split=split_name)

        return metrics

    def save_gradcam(self, result, result_dir, filename, remove_duplicate=""):
        result_file = os.path.join(result_dir, '%s_rank%d.pth' % (filename, get_rank()))
        final_result_file = os.path.join(result_dir, '%s.pth' % filename)
        torch.save({'result': result}, result_file)

        dist.barrier()

        if is_main_process():
            logging.warning("rank %d starts merging results." % get_rank())
            # combine results from all processes
            result = []

            for rank in range(get_world_size()):
                result_file = os.path.join(result_dir, '%s_rank%d.pth' % (filename, rank))
                res_ckpt = torch.load(result_file, map_location='cpu')
                res = res_ckpt['result']

                result += res

            if remove_duplicate:
                result_new = []
                id_list = []
                for res in result:
                    if res[remove_duplicate] not in id_list:
                        id_list.append(res[remove_duplicate])
                        result_new.append(res)
                result = result_new

            torch.save({'result': result}, final_result_file)
            print("result file saved to %s" % final_result_file)

        return final_result_file


@registry.register_task("gqa_reading_comprehension")
class GQARCTask(VQARCTask):
    def valid_step(self, model, samples):
        answers, captions, gradcams = model.predict_answers(
            samples=samples,
            inference_method=self.inference_method,
            num_beams=self.num_beams,
            max_len=self.max_len,
            min_len=self.min_len,
            internal_bsz_fid=self.config['internal_bsz_fid'],
            num_captions=self.config['num_captions'],
            num_captions_fid=self.config['num_captions_fid'],
            cap_max_length=self.config['cap_max_length'],
            cap_min_length=self.config['cap_min_length'],
            top_k=self.config['top_k'],
            top_p=self.config['top_p'],
            repetition_penalty=self.config['repetition_penalty'],
            num_patches=self.config['num_patches'],
            block_num=self.config['block_num'],
        )

        pred_qa_pairs = []
        sample_captions = []
        sample_gradcams = []

        question_id = samples["question_id"]
        gt_answers = samples["answer"]

        for pred_answer, caption, gradcam, ques_id, gt_answer in zip(answers, captions, gradcams, question_id, gt_answers):
            ques_id = int(ques_id.item())
            pred_qa_pairs.append({"question_id": ques_id, "pred_ans": pred_answer, "gt_ans": gt_answer})
            sample_captions.append({"question_id": ques_id, "caption": caption})
            sample_gradcams.append({"question_id": ques_id, "gradcam": gradcam})

        return [sample_gradcams, sample_captions, pred_qa_pairs]

    @dist_utils.main_process
    def _report_metrics(self, result_file, split):
        """
        TODO: add other evaluation metrics for GQA
        """

        results = json.load(open(result_file, "r"))
        acc = []
        vqa_tool = VQATool()

        for res in results:
            if res["gt_ans"] is None:
                # prepare test results for leaderboard evaluation
                self._save_result_leaderboard(results)
                return

            gt_ans = res["gt_ans"]
            pred = res["pred_ans"]

            if self.inference_method == "generate":
                pred = vqa_tool.processPunctuation(pred)
                pred = vqa_tool.processDigitArticle(pred)

            vqa_acc = 1 if pred == gt_ans else 0

            acc.append(vqa_acc)

        accuracy = sum(acc) / len(acc) * 100
        metrics = {"agg_metrics": accuracy, "acc": accuracy}

        with open(
            os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
        ) as f:
            f.write(json.dumps(metrics) + "\n")

        logging.info(metrics)

        return metrics

    @dist_utils.main_process
    def _save_result_leaderboard(self, results):
        """
        Saving the results in the format required for leaderboard evaluation.
        """
        result_leaderboard = []
        for res in results:
            result_leaderboard.append({
                "questionId": str(res['question_id']),
                "prediction": str(res["pred_ans"]),
            })

        result_file = registry.get_path("result_dir") + "_leaderboard.json"

        with open(result_file, "w") as f:
            json.dump(result_leaderboard, f)

        logging.info(f"Saved results for leaderboard evaluation at {result_file}")
