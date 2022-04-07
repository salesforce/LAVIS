import json
import logging
import os

import torch.distributed as dist
import utils.blip_utils as utils
from common.registry import registry

from tasks.base_task import BaseTask


@registry.register_task('vqa')
class VQATask(BaseTask):
    def __init__(self, num_beams, max_len, min_len, evaluate, num_ans_candidates, inference_method="rank"):
        super().__init__()

        self.num_beams = num_beams
        self.max_len = max_len
        self.min_len = min_len

        self.evaluate = evaluate
        self.inference_method = inference_method
        self.num_ans_candidates = num_ans_candidates

        self.answer_list = None

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg

        num_beams = run_cfg.get("num_beams", 3)
        max_len = run_cfg.get("max_len", 10)
        min_len= run_cfg.get("min_len", 1)

        evaluate = run_cfg.get("evaluate", False)

        inference_method = run_cfg.get("inference_method", "rank")
        num_ans_candidates = run_cfg.get("num_ans_candidates", 128)

        return cls(
            num_beams=num_beams, 
            max_len=max_len, 
            min_len=min_len,
            evaluate=evaluate,
            num_ans_candidates=num_ans_candidates,
            inference_method=inference_method
        )

    def build_datasets(self, cfg):
        datasets = super().build_datasets(cfg)

        assert 'test' in datasets, "No testing split is present."
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
            num_ans_candidates=self.num_ans_candidates
        )

        pred_qa_pairs = [] 

        question_id = samples['question_id']
        for answer, ques_id in zip(answers, question_id):
            ques_id = int(ques_id.item())       
            pred_qa_pairs.append({"question_id": ques_id, "answer": answer})             

        return pred_qa_pairs

    def after_validation(self, val_result, result_dir, **kwargs):
        save_result(val_result, result_dir, filename='vqa_result')
        return val_result


def save_result(result, result_dir, filename, remove_duplicate=''):
    result_file = os.path.join(result_dir, '%s_rank%d.json'%(filename,utils.get_rank()))
    final_result_file = os.path.join(result_dir, '%s.json'%filename)
    
    json.dump(result,open(result_file,'w'))

    dist.barrier()

    if utils.is_main_process():   
        # combine results from all processes
        result = []

        for rank in range(utils.get_world_size()):
            result_file = os.path.join(result_dir, '%s_rank%d.json'%(filename,rank))
            res = json.load(open(result_file,'r'))
            result += res

        if remove_duplicate:
            result_new = []
            id_list = []    
            for res in result:
                if res[remove_duplicate] not in id_list:
                    id_list.append(res[remove_duplicate])
                    result_new.append(res)
            result = result_new             
                
        json.dump(result,open(final_result_file,'w'))            
        logging.info('result file saved to %s'%final_result_file)

    return final_result_file
