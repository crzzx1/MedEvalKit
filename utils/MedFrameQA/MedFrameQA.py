import torch
import os
import json
import gc
from PIL import Image
from datasets import load_dataset
from collections import defaultdict
from tqdm import tqdm

from ..utils import save_json, extract, judge_multi_choice
from ..base_dataset import BaseDataset
from ..question_formats import get_multiple_choice_prompt

class MedFrameQA(BaseDataset):
    def __init__(self, model, dataset_path, output_path):
        super().__init__() # 调用父类的 __init__
        self.model = model
        self.output_path = output_path
        self.dataset_path = dataset_path if dataset_path else "SuhaoYu1020/MedFrameQA"
        # self.samples 列表不再需要了，因为我们采用流式处理
        self.chunk_idx = int(os.environ.get("chunk_idx", 0))
        self.num_chunks = int(os.environ.get("num_chunks", 1))

    # ❶ load_data 被修改为生成器 (Generator)
    def load_data(self):
        """
        这个方法现在是一个生成器，它会逐个样本'yield'出来，而不是全部加载到内存。
        """
        print("正在以“生成器”模式加载 MedFrameQA 数据集...")
        dataset = load_dataset(self.dataset_path)["test"]
        for idx, sample in enumerate(dataset):
            if idx % self.num_chunks == self.chunk_idx:
                # 处理并'生产'一个样本
                yield self.construct_messages(sample)

    # construct_messages 方法保持不变，因为它处理单个样本
    def construct_messages(self, sample):
        question = sample["question"]
        choices = sample["options"]
        answer = sample["correct_answer"]
        image_1 = sample["image_1"]
        image_2 = sample["image_2"]
        image_3 = sample["image_3"]
        image_4 = sample["image_4"]
        image_5 = sample["image_5"]

        choices_formatted = [f"{chr(65+i)}.{choices[i]}" for i in range(len(choices))]
        images = [img for img in [image_1, image_2, image_3, image_4, image_5] if img is not None]

        is_reasoning = True if os.environ.get("REASONING", "False") == "True" else False
        prompt = get_multiple_choice_prompt(question, choices_formatted, is_reasoning)
        
        # 将原始图像对象直接放入 messages
        messages = {"prompt": prompt, "images": images}
        
        # 为评估准备数据
        sample["messages"] = messages
        sample["choices"] = choices_formatted
        sample["answer"] = answer
        
        # 清理不再需要的原始数据以节省内存
        del sample["options"], sample["correct_answer"]
        del sample["image_1"], sample["image_2"], sample["image_3"], sample["image_4"], sample["image_5"]
        
        return sample

    # ❷ 新增一个内存高效的 eval 方法
    def eval(self):
        """
        这个方法覆盖了父类的方法，实现了基于生成器的分批推理流程。
        """
        out_samples = []
        batch_size = 16  # 可以根据您的GPU显存大小调整
        batch = []
        
        # self.load_data() 返回一个生成器，我们在这里遍历它
        # 整个数据集不会同时存在于内存中
        print(f"开始分批推理，批大小: {batch_size}...")
        for sample in tqdm(self.load_data()):
            batch.append(sample)
            if len(batch) >= batch_size:
                # 当批次满了，就进行一次模型推理
                messages_batch = [s["messages"] for s in batch]
                outputs = self.model.generate_outputs(messages_batch)
                
                # 处理批次结果
                for i, response in enumerate(outputs):
                    s = batch[i]
                    del s["messages"] # 推理后删除 messages 节约内存
                    s["response"] = response
                    out_samples.append(s)
                
                batch = [] # 清空批次，准备下一批
                gc.collect()

        # 处理最后一个可能未满的批次
        if batch:
            messages_batch = [s["messages"] for s in batch]
            outputs = self.model.generate_outputs(messages_batch)
            for i, response in enumerate(outputs):
                s = batch[i]
                del s["messages"]
                s["response"] = response
                out_samples.append(s)
            gc.collect()

        print("所有样本推理完成，开始计算指标...")
        # 所有样本都处理完后，再进行指标计算
        metrics, out_samples = self.cal_metrics(out_samples)
        
        # 保存结果文件
        dataset_name = self.dataset_path.split("/")[-1]
        save_json_path = os.path.join(self.output_path, f"{dataset_name}.json")
        save_json(save_json_path, out_samples)

        return {"final results on {}".format(dataset_name): metrics}

    # cal_metrics 方法保持不变
    def cal_metrics(self, out_samples):
        total = 0
        right = 0
        for i, sample in enumerate(out_samples):
            response = sample["response"]
            choices = sample["choices"]
            answer = sample["answer"]
            correct = judge_multi_choice(choices, answer, response)
            out_samples[i]["correct"] = correct
            if correct:
                right += 1
            total += 1
        metrics = {"total metrics": {"total": total, "right": right, "acc": right / total}}
        return metrics, out_samples