import torch
import os
import json
import gc
import csv

from PIL import Image
from datasets import load_dataset
from collections import defaultdict
from tqdm import tqdm

from ..utils import save_json, extract, judge_multi_choice
from ..base_dataset import BaseDataset
from ..question_formats import get_multiple_choice_prompt

medical_subject = ["anatomy","clinical_knowledge","college_biology","college_medicine","medical_genetics","professional_medicine","nutrition","virology","high_school_biology"]

class CMB(BaseDataset):
    def __init__(self, model, dataset_path, output_path):
        self.model = model
        self.output_path = output_path
        self.dataset_path = dataset_path if dataset_path else "FreedomIntelligence/CMB"
        # --- 使用您验证过的正确文件路径 ---
        self.question_file = "hf://datasets/FreedomIntelligence/CMB/CMB-Exam/CMB-test/CMB-test-choice-question-merge.json"
        
        # --- 答案文件路径修正为 GitHub 的原始文件链接 ---
        self.answer_file = "https://raw.githubusercontent.com/FreedomIntelligence/CMB/main/data/CMB-test-choice-answer.json"
        # --- 路径修正完毕 ---

        self.samples = []
        self.chunk_idx = int(os.environ.get("chunk_idx", 0))
        self.num_chunks = int(os.environ.get("num_chunks", 1))

    def load_data(self):
        print(f"正在加载测试集问题: {self.question_file}")
        question_dataset = load_dataset("json", data_files=self.question_file, split="train")
        
        print(f"正在加载测试集答案: {self.answer_file}")
        answer_dataset = load_dataset("json", data_files=self.answer_file, split="train")

        answer_map = {sample['id']: sample['answer'] for sample in answer_dataset}
        print(f"成功加载了 {len(answer_map)} 条答案。")

        for idx, sample in tqdm(enumerate(question_dataset), total=len(question_dataset)):
            if idx % self.num_chunks != self.chunk_idx:
                continue
            
            if sample.get("question_type") != "单项选择题":
                continue
            
            if sample['id'] in answer_map:
                sample['answer'] = answer_map[sample['id']]
                self.samples.append(self.construct_messages(sample))
            else:
                print(f"警告：未能为 ID 为 {sample['id']} 的样本找到答案。")
                
        return self.samples

    def construct_messages(self, sample):
        # 将变量重命名为 option_dict 更准确地反映其类型
        option_dict = sample["option"]
        question = sample["question"]
        
        # 使用 .items() 方法正确地遍历字典的键和值
        choices = [f"{key}. {value}" for key, value in option_dict.items() if value]

        is_reasoning = True if os.environ.get("REASONING", "False") == "True" else False
        prompt = get_multiple_choice_prompt(question, choices, is_reasoning, lang="zh")
        
        messages = {"prompt": prompt}
        sample["prompt"] = prompt
        sample["messages"] = messages
        sample["choices"] = choices
        return sample

    def cal_metrics(self, out_samples):
        total = 0
        right = 0
        for i, sample in enumerate(out_samples):
            response = sample["response"]
            response = extract(response, "answer")
            choices = sample["choices"]
            answer = sample["answer"]

            correct = judge_multi_choice(choices, answer, response)
            out_samples[i]["correct"] = correct
            if correct:
                right += 1
            total += 1

        metrics = {"total metrics": {"total": total, "right": right, "acc": right / total}}
        return metrics, out_samples