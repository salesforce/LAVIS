"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.common.registry import registry
from lavis.tasks.base_task import BaseTask
# from lavis.tasks.captioning import valid_step as valid_step_captioning, coco_caption_eval

@registry.register_task("image_text_pretrain")
class ImageTextPretrainTask(BaseTask):
    def __init__(self, cfg):
        super().__init__()
        self.num_beams = num_beams
        self.max_len = max_len
        self.min_len = min_len
        self.evaluate = evaluate
        self.cfg = cfg

        self.report_metric = report_metric

#     @classmethod
#     def setup_task(cls, cfg):
#         run_cfg = cfg.run_cfg

#         num_beams = run_cfg.num_beams
#         max_len = run_cfg.max_len
#         min_len = run_cfg.min_len
#         evaluate = run_cfg.evaluate

#         report_metric = run_cfg.get("report_metric", True)

#         return cls(
#             num_beams=num_beams,
#             max_len=max_len,
#             min_len=min_len,
#             evaluate=evaluate,
#             cfg=run_cfg,
#             report_metric=report_metric,
#         )
    
#     def evaluation(self, model, data_loader, cuda_enabled=True, **kwargs):
#         # retrieval evaluation
#         score_i2t, score_t2i = model.compute_sim_matrix(data_loader, task_cfg=self.cfg)
        
#         retrieval_result = self._compute_recall(score_i2t,score_t2i,data_loader.dataset.txt2img,data_loader.dataset.img2txt)
        
#         # caption evaluation
#         metric_logger = MetricLogger(delimiter="  ")
#         header = "Captioning Evaluation"
#         # TODO make it configurable
#         print_freq = 10

#         caption_result = []

#         for samples in metric_logger.log_every(data_loader, print_freq, header):
#             samples = prepare_sample(samples, cuda_enabled=cuda_enabled)

#             eval_output = self.valid_step(model=model, samples=samples)
#             caption_result.extend(eval_output)

#         if is_dist_avail_and_initialized():
#             dist.barrier()    

#         return retrieval_result, caption_result
    

#     def after_evaluation(self, retrieval_result, caption_result, **kwargs):
   
#         caption_result_file = self.save_result(
#             result=caption_result,
#             result_dir=registry.get_path("result_dir"),
#             filename="{}_epoch{}".format(split_name, epoch),
#             remove_duplicate="image_id",
#         )            

#         if is_main_process():
#             eval_result = self._report_metrics(
#                 retrieval_result,
#                 caption_result_file, 
#                 split_name,
#             )
#             logging.info(eval_result)
#         else:
#             eval_result = None
            
#         return val_result
    
    
#     @staticmethod
#     @torch.no_grad()
#     def _compute_recall(scores_i2t, scores_t2i, txt2img, img2txt):

#         # Images->Text
#         ranks = np.zeros(scores_i2t.shape[0])
#         for index, score in enumerate(scores_i2t):
#             inds = np.argsort(score)[::-1]
#             # Score
#             rank = 1e20
#             for i in img2txt[index]:
#                 tmp = np.where(inds == i)[0][0]
#                 if tmp < rank:
#                     rank = tmp
#             ranks[index] = rank

#         # Compute metrics
#         tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
#         tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
#         tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

#         # Text->Images
#         ranks = np.zeros(scores_t2i.shape[0])

#         for index, score in enumerate(scores_t2i):
#             inds = np.argsort(score)[::-1]
#             ranks[index] = np.where(inds == txt2img[index])[0][0]

#         # Compute metrics
#         ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
#         ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
#         ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

#         tr_mean = (tr1 + tr5 + tr10) / 3
#         ir_mean = (ir1 + ir5 + ir10) / 3
#         r_mean = (tr_mean + ir_mean) / 2
        
#         eval_result = {
#             "txt_r1": tr1,
#             "txt_r_mean": tr_mean,
#             "img_r1": ir1,
#             "img_r_mean": ir_mean,
#             "r_mean": r_mean,
            
#         }                    
#         return eval_result
        

#     @main_process
#     @torch.no_grad()
#     def _report_metrics(retrieval_result, caption_result_file, caption_split_name):

#         coco_gt_root = os.path.join(registry.get_path("cache_root"), "coco_gt")
#         coco_val = coco_caption_eval(coco_gt_root, caption_result_file, caption_split_name)

#         eval_result = {
#             **{f'{k}': v for k, v in retrieval_result.items()}
#             **{f'{k}': v for k, v in coco_val.eval.items()}
            
#         }        

#         with open(
#             os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
#         ) as f:
#             f.write(json.dumps(eval_result) + "\n")

#         return eval_result
    
    