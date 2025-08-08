import torch
import os
import json
import gc
import csv

from PIL import Image
from datasets import load_dataset,load_from_disk
from collections import defaultdict
from tqdm import tqdm

from ..utils import save_json,extract,judge_multi_choice
from ..base_dataset import BaseDataset

from ..question_formats import get_multiple_choice_prompt
medical_subject = ["anatomy","clinical_knowledge","college_medicine","genetics","traditional_chinese_medicine","nutrition","virology"]

class CMMLU(BaseDataset):
    def __init__(self,model,dataset_path,output_path):
        self.model = model
        self.output_path = output_path
        self.dataset_path = dataset_path if dataset_path else "haonan-li/cmmlu"
        self.samples = []
        self.chunk_idx = int(os.environ.get("chunk_idx",0))
        self.num_chunks = int(os.environ.get("num_chunks",1))
    
    def load_data(self):
        # 遍历您在文件顶部定义的 medical_subject 列表
        for subject in medical_subject:
            print(f"正在加载 CMMLU 的子集: {subject}...")
            # 逐个加载每个主题的数据集
            # 注意：我们将主题名(subject)作为第二个参数传给 load_dataset
            # 同时，CMMLU 的评估通常在 "test" split 上进行
            try:
                dataset = load_dataset(self.dataset_path, name=subject, split="test", trust_remote_code=True)
            except Exception as e:
                print(f"加载子集 {subject} 失败: {e}。跳过此子集。")
                continue

            for idx, sample in tqdm(enumerate(dataset), desc=f"处理 {subject}"):
                if idx % self.num_chunks == self.chunk_idx:
                    # 这里的 if sample["task"] in medical_subject 检查其实可以省略了
                    # 因为我们已经是按主题加载的，但保留也无妨
                    sample = self.construct_messages(sample)
                    self.samples.append(sample)

        print(f"成功加载了 {len(self.samples)} 条来自 CMMLU 医学主题的样本。")
        return self.samples

    def construct_messages(self,sample):
        # 确保这一行以及方法内的所有行都有正确的缩进（通常是4个空格）
        answer = sample["Answer"]

        OptionA = sample["A"]
        OptionB = sample["B"]
        OptionC = sample["C"]
        OptionD = sample["D"]
        question = sample["Question"]

        choices = [OptionA,OptionB,OptionC,OptionD]
        choices = [f"{chr(65+i)}.{choices[i]}" for i in range(len(choices))]

        is_reasoning = True if os.environ.get("REASONING","False") == "True" else False
        prompt = get_multiple_choice_prompt(question,choices,is_reasoning,lang = "zh")

        messages = {"prompt":prompt}
        sample["prompt"] = prompt
        sample["messages"] = messages
        sample["answer"] = answer 
        sample["choices"] = choices
        return sample


    def cal_metrics(self,out_samples):
        total = 0
        right = 0
        for i,sample in enumerate(out_samples):
            response = sample["response"]
            response = extract(response,"answer")
            choices = sample["choices"]
            answer = sample["answer"]

            correct = judge_multi_choice(choices,answer,response)
            out_samples[i]["correct"] = correct
            if correct:
                right += 1
            total += 1

        metrics = {"total metrics":{"total":total,"right":right,"acc":right/total}}
        return metrics,out_samples

