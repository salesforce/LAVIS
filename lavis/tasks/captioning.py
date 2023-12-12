"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import json
import os
import pandas as pd
from tqdm import tqdm
from lavis.common.dist_utils import main_process, get_rank
from lavis.common.registry import registry
from lavis.tasks.base_task import BaseTask
from lavis.common.utils import is_convertible_to_int, is_url, cache_url

@registry.register_task("captioning")
class CaptionTask(BaseTask):
    def __init__(self, num_beams, max_len, min_len, repetition_penalty, length_penalty, top_p, temperature, evaluate, report_metric=True, annotation_file=None, sample_id_key="image_id", caption_key="caption", split=["val"], load_gt_from_file=False, img_ids = []):
        super().__init__()

        self.num_beams = num_beams
        self.max_len = max_len
        self.min_len = min_len
        self.repetition_penalty = repetition_penalty
        self.length_penalty = length_penalty
        self.top_p = top_p
        self.temperature = temperature
        self.evaluate = evaluate

        self.report_metric = report_metric
        self.annotation_file = annotation_file
        self.sample_id_key = sample_id_key
        self.caption_key = caption_key
        assert len(split) == 1, "Only support one split for evaluation."
        self.split = split[0]
        self.load_gt_from_file = load_gt_from_file
        self.img_ids = img_ids

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg

        num_beams = run_cfg.get("num_beams", 5)
        max_len = run_cfg.get("max_len", 30)
        min_len = run_cfg.get("min_len", 1)
        repetition_penalty = run_cfg.get("repetition_penalty", 1.15)
        length_penalty = run_cfg.get("length_penalty", 0.)
        top_p = run_cfg.get("top_p", 0.9)
        temperature = run_cfg.get("temperature", 1.)
        evaluate = run_cfg.evaluate

        report_metric = run_cfg.get("report_metric", True)
        annotation_file = run_cfg.get("annotation_file", None)
        sample_id_key = run_cfg.get("sample_id_key", "image_id")
        caption_key = run_cfg.get("caption_key", "caption")
        load_gt_from_file = run_cfg.get("load_gt_from_file", False)
        split = run_cfg.get("valid_splits", ["val"])
        img_ids = run_cfg.get("img_ids", []) # evaluate only subset of imgs

        return cls(
            num_beams=num_beams,
            max_len=max_len,
            min_len=min_len,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            top_p=top_p,
            temperature=temperature,
            evaluate=evaluate,
            report_metric=report_metric,
            annotation_file=annotation_file,
            sample_id_key=sample_id_key,
            caption_key=caption_key,
            split=split,
            load_gt_from_file=load_gt_from_file,
            img_ids=img_ids
        )

    def build_datasets(self, cfg):
        datasets = super().build_datasets(cfg)

        # get validation dataset name
        val_ds_name = []
        for name,d in datasets.items():
            if self.split in d:
                val_ds_name.append(name)
        if not val_ds_name:
            return datasets # no validation sets
        assert len(val_ds_name) == 1, "Only support one dataset for validation"
        val_ds_name = val_ds_name[0]

        # get question file, annotation file and anwser list in COCO format
        if self.annotation_file == None:
            if 'coco' not in val_ds_name: # coco is already precomputed in dataset
                self.annotation_file = os.path.join(registry.get_path("cache_root"),f'{val_ds_name}_gt', f'{val_ds_name}_{self.split}_annotations.json')
                if get_rank() == 0:
                    os.makedirs(os.path.join(registry.get_path("cache_root"),f'{val_ds_name}_gt'), exist_ok=True)
                    convert_to_coco_gt(datasets[val_ds_name], self.annotation_file, self.caption_key, self.sample_id_key, self.split, load_gt_from_file=self.load_gt_from_file, img_ids=self.img_ids)
        return datasets

    def valid_step(self, model, samples):
        results = []
        # run_cfg = slf.cfg.run_cfg
        captions = model.generate(
            samples,
            use_nucleus_sampling=False,
            num_beams=self.num_beams,
            max_length=self.max_len,
            min_length=self.min_len,
            repetition_penalty=self.repetition_penalty,
            length_penalty=self.length_penalty,
            top_p=self.top_p,
            temperature=self.temperature,
        )
        img_ids = samples[self.sample_id_key]
        for caption, img_id in zip(captions, img_ids):
            # not all img_ids are ints
            img_id = int(img_id) if is_convertible_to_int(img_id) else img_id
            if self.img_ids and img_id not in self.img_ids: # only include specified img_ids if specified
                continue
            results.append({"caption": caption, "image_id": img_id})

        return results

    def after_evaluation(self, val_result, split_name, epoch, **kwargs):
        eval_result_file = self.save_result(
            result=val_result,
            result_dir=registry.get_path("result_dir"),
            filename="{}_epoch{}".format(split_name, epoch),
            remove_duplicate="image_id",
        )

        if self.report_metric:
            metrics = self._report_metrics(
                eval_result_file=eval_result_file, split_name=split_name
            )
        else:
            metrics = {"agg_metrics": 0.0}

        return metrics

    @main_process
    def _report_metrics(self, eval_result_file, split_name):

        if self.annotation_file == None:
            # TODO better way to define this
            coco_gt_root = os.path.join(registry.get_path("cache_root"), "coco_gt")
            coco_val = coco_caption_eval(coco_gt_root, eval_result_file, split_name, img_ids=self.img_ids)
        else:
            coco_val = coco_caption_eval(None, eval_result_file, split_name, annotation_file=self.annotation_file, img_ids=self.img_ids)

        agg_metrics = coco_val.eval["CIDEr"] + coco_val.eval["Bleu_4"]
        log_stats = {split_name: {k: v for k, v in coco_val.eval.items()}}

        with open(
            os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
        ) as f:
            f.write(json.dumps(log_stats) + "\n")

        coco_res = {k: v for k, v in coco_val.eval.items()}
        coco_res["agg_metrics"] = agg_metrics

        return coco_res

def load_gt_file(file_path):
    if is_url(file_path):
        file_path = cache_url(file_path, registry.get_path("cache_root"))
    data = []
    if any(ext in file_path for ext in ['csv', 'tsv']):
        df = pd.read_csv(file_path)
        data.extend(df.to_dict(orient="records"))
        
    elif 'jsonl' in file_path:
        with open(file_path, "r") as f:
            data.extend([json.loads(line) for line in f])
    else:
        with open(file_path, "r") as f:
            loaded = json.load(f)
            if isinstance(loaded, list):
                data.extend(loaded)
            elif isinstance(loaded, dict):
                # assume that loaded data in file  is the corresponding caption to the key
                data.extend([{"sample_id": k, **v} if isinstance(v, dict) else {"sample_id": k, "caption": v} for k, v in loaded.items()])
    return data

def convert_to_coco_gt(data, outpath, caption_key, sample_id_key, split, load_gt_from_file=False, img_ids=[]):
    gt_data = {"annotations":[], "images":[]}
    if load_gt_from_file:
        print(f"Generating ground truth file for evaluation from {load_gt_from_file}....")
        data = load_gt_file(load_gt_from_file)
        for ann in data:
            captions = ann[caption_key]
            img_id = int(ann[sample_id_key]) if is_convertible_to_int(ann[sample_id_key]) else ann[sample_id_key]
            if img_ids and img_id not in img_ids: # only include specified img_ids if specified
                continue
            gt_data["images"].append({"id":img_id})
            if isinstance(captions, str):
                gt_data["annotations"].append({"image_id":img_id, "caption":captions, "id":img_id})
            else:   
                gt_data["annotations"].extend([{"image_id":img_id, "caption":c, "id":img_id} for c in captions])
    else:
        print(f"Generating ground truth file for evaluation....")
        for i,ann in tqdm(enumerate(data[split])):
            captions = data[split].annotation[i][caption_key]
            img_id = int(ann[sample_id_key]) if is_convertible_to_int(ann[sample_id_key]) else ann[sample_id_key]
            if img_ids and img_id not in img_ids: # only include specified img_ids if specified
                continue
            gt_data["images"].append({"id":img_id})
            if isinstance(captions, str):
                gt_data["annotations"].append({"image_id":img_id, "caption":captions, "id":img_id})
            else:   
                gt_data["annotations"].extend([{"image_id":img_id, "caption":c, "id":img_id} for c in captions])
    json.dump(gt_data, open(outpath, 'w'))
    print(f"Saved annotations at {outpath}")

# TODO better structure for this.
from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO
from torchvision.datasets.utils import download_url


def coco_caption_eval(coco_gt_root, results_file, split, annotation_file=None, img_ids=[]):

    if annotation_file == None:
        urls = {
            "val": "https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val_gt.json",
            "test": "https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test_gt.json",
        }
        filenames = {
            "val": "coco_karpathy_val_gt.json",
            "test": "coco_karpathy_test_gt.json",
        }

        download_url(urls[split], coco_gt_root)
        annotation_file = os.path.join(coco_gt_root, filenames[split])
    if is_url(annotation_file):
        annotation_file = cache_url(annotation_file, registry.get_path("cache_root"))
        
    # create coco object and coco_result object
    coco = COCO(annotation_file)
    coco_result = coco.loadRes(results_file)

    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)

    # evaluate on a subset of images by setting
    if img_ids:
        coco_eval.params['image_id'] = coco_result.getImgIds()
    # please remove this line when evaluating the full validation set
    # coco_eval.params['image_id'] = coco_result.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    coco_eval.evaluate()

    # print output evaluation scores
    for metric, score in coco_eval.eval.items():
        print(f"{metric}: {score:.3f}")

    return coco_eval
