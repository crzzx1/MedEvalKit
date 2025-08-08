# /home/ubuntu/MedEvalKit/utils/PATH_VQA/PATH_VQA.py
# (请用以下全部内容覆盖原文件)

import torch
import os
import json
from PIL import Image
from datasets import load_dataset
from tqdm import tqdm
from mathruler.grader import extract_boxed_content

from ..utils import save_json,extract,judger,get_compare_messages,judge_open_end_vqa,judge_judgement,judge_close_end_vqa
from ..base_dataset import BaseDataset
from ..question_formats import get_pathvqa_judgement_prompt, get_pathvqa_open_prompt

class PATH_VQA(BaseDataset):
    def __init__(self,model,dataset_path,output_path):
        self.model = model
        self.output_path = output_path
        self.dataset_path = dataset_path if dataset_path else "flaviagiammarino/path-vqa"
        self.samples = []
        self.chunk_idx = int(os.environ.get("chunk_idx",0))
        self.num_chunks = int(os.environ.get("num_chunks",1))

    def load_data(self):
        dataset = load_dataset(self.dataset_path, split="test")
        for idx, sample in tqdm(enumerate(dataset)):
            if idx % self.num_chunks == self.chunk_idx:
                self.samples.append(self.construct_messages(sample))
        return self.samples

    def construct_messages(self,sample):
        question = sample["question"]
        image = sample["image"]
        answer = sample["answer"]
        is_reasoning = True
        
        if answer.lower() in ["yes","no"]:
            prompt = get_pathvqa_judgement_prompt(question, is_reasoning)
        else:
            prompt = get_pathvqa_open_prompt(question, is_reasoning)

        sample["messages"] = {"prompt": prompt, "image": image}
        if "image" in sample: del sample["image"]
        return sample

    def cal_metrics(self,out_samples):
        messages_list = []
        metrics = {
            "total metrics": {"total": 0, "right": 0},
            "open": {"total": 0, "right": 0, "bleu1": 0, "bleu2": 0, "bleu3": 0, "bleu4": 0, "rouge1": 0, "rouge2": 0, "rougel": 0, "precision": 0, "recall": 0, "f1": 0, "em": 0},
            "close": {"total": 0, "right": 0}
        }
        open_id = []

        for i, out_sample in tqdm(enumerate(out_samples)):
            response = out_sample["response"]
            if extract_boxed_content(response) != "None":
                response = extract_boxed_content(response)
            elif "<answer>" in response:
                response = extract(response, "answer")

            answer = out_sample["answer"].lower().strip()
            question = out_sample["question"]
            response = response.lower().strip()

            metrics["total metrics"]["total"] += 1
            if answer in ["yes", "no"]:
                metrics["close"]["total"] += 1
                correct = judge_judgement(answer, response)
                out_samples[i]["correct"] = correct
                if correct:
                    metrics["close"]["right"] += 1
                    metrics["total metrics"]["right"] += 1
            else:
                metrics["open"]["total"] += 1
                c_metrics = judge_open_end_vqa(answer, response)
                out_samples[i]["correct"] = c_metrics["em"]
                if c_metrics["em"]:
                    metrics["total metrics"]["right"] += 1
                    metrics["open"]["right"] += 1
                for metric in c_metrics:
                    metrics["open"][metric] += c_metrics[metric]
                if os.environ.get("use_llm_judge", "False") == "True":
                    messages = get_compare_messages(question, response, answer)
                    messages_list.append(messages)
                    open_id.append(i)

        if os.environ.get("use_llm_judge", "False") == "True":
            original_open_em_right = metrics["open"]["right"]
            metrics["total metrics"]["right"] -= original_open_em_right
            metrics["open"]["right"] = 0
            
            llm = judger
            results = llm.generate_outputs(messages_list)
            
            for i, result in zip(open_id, results):
                if result is None: continue
                result = extract(result, "judge")
                is_correct = True if result == "0" else False
                out_samples[i]["correct"] = is_correct
                if is_correct:
                    metrics["open"]["right"] += 1
            
            metrics["total metrics"]["right"] += metrics["open"]["right"]

        # Calculate final accuracies
        for category in ["total metrics", "open", "close"]:
            if metrics[category]["total"] > 0:
                metrics[category]["acc"] = metrics[category]["right"] / metrics[category]["total"]
            else:
                metrics[category]["acc"] = 0
        
        # Calculate other average metrics for open questions
        if metrics["open"]["total"] > 0:
            for metric in metrics["open"]:
                if metric not in ["right", "total", "acc"]:
                    metrics["open"][metric] /= metrics["open"]["total"]

        return metrics, out_samples