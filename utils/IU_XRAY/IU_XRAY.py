import torch
import os
import json
import gc
import csv

from PIL import Image
from datasets import load_dataset
from collections import defaultdict
from tqdm import tqdm

import numpy as np

from ..utils import save_json,extract
from ..base_dataset import BaseDataset

from ..question_formats import get_report_generation_prompt

class IU_XRAY(BaseDataset):
    def __init__(self,model,dataset_path,output_path):
        self.model = model
        self.output_path = output_path
        self.dataset_path = dataset_path
        self.samples = []
        self.chunk_idx = int(os.environ.get("chunk_idx",0))
        self.num_chunks = int(os.environ.get("num_chunks",1))

    def load_data(self):
        dataset_path = self.dataset_path
        json_path = os.path.join(dataset_path,"test.json")

        # 使用 utf-8 编码打开文件以增加兼容性
        with open(json_path,"r", encoding='utf-8') as f:
            dataset = json.load(f)

        for idx,sample in tqdm(enumerate(dataset), desc="Loading IU_XRAY data"):
            if idx % self.num_chunks == self.chunk_idx:
                # 过滤掉没有报告内容的样本
                if not sample.get("findings", "").strip() and not sample.get("impression", "").strip():
                    continue
                sample = self.construct_messages(sample)
                self.samples.append(sample)
                
        print("Total samples number:", len(self.samples))
        return self.samples

# ========================================================================
# === 请用这个新函数完整替换掉 IU_XRAY.py 中的 construct_messages 函数 ===
# ========================================================================

# ========================================================================
# === 请用这个新函数完整替换掉 IU_XRAY.py 中的 construct_messages 函数 ===
# ========================================================================

    def construct_messages(self, sample):
        # 基础的图片根目录
        image_root = os.path.join(self.dataset_path, "images")
        
        # 1. 从 sample 中获取图片文件名列表 (例如 ["CXR1124_IM-0081-2001.png", "CXR1124_IM-0081-3001.png"])
        image_filenames_from_json = sample.get("image", [])
        if isinstance(image_filenames_from_json, str):
            image_filenames_from_json = [image_filenames_from_json]
            
        if not image_filenames_from_json:
            return sample # 如果没有图片信息，直接返回

        # 2. 根据列表中的第一个文件名，确定图片所在的子文件夹名
        #    例如 "CXR1124_IM-0081-2001.png" -> "CXR1124_IM-0081"
        sub_dir_name = image_filenames_from_json[0].rsplit('-', 1)[0]
        sub_dir_path = os.path.join(image_root, sub_dir_name)

        if not os.path.isdir(sub_dir_path):
            raise FileNotFoundError(f"找不到图片子文件夹: {sub_dir_path}")

        # 3. 获取该子文件夹中所有真实图片文件，并按字母顺序排序
        #    这会得到一个像 ['0.png', '1.png'] 这样的有序列表
        actual_files_in_dir = sorted([
            f for f in os.listdir(sub_dir_path) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

        # 4. 按顺序加载图片：json 列表中的第 N 个，对应文件夹中的第 N 张
        pil_images = []
        num_images_to_load = len(image_filenames_from_json)
        
        if len(actual_files_in_dir) < num_images_to_load:
            raise ValueError(
                f"JSON 中需要 {num_images_to_load} 张图片, "
                f"但在文件夹 {sub_dir_path} 中只找到了 {len(actual_files_in_dir)} 张。"
            )

        for i in range(num_images_to_load):
            image_to_load_name = actual_files_in_dir[i]
            full_path = os.path.join(sub_dir_path, image_to_load_name)
            pil_images.append(Image.open(full_path))

        # --- 后续逻辑保持不变 ---
        findings = sample.get("findings", "")
        impression = sample.get("impression", "")
        findings = "None" if not findings.strip() else findings
        impression = "None" if not impression.strip() else impression
        
        prompt = get_report_generation_prompt()

        messages = {"prompt": prompt, "images": pil_images}
        sample["messages"] = messages
        return sample


    def cal_metrics(self,out_samples):
        import pandas as pd

        predictions_data = []
        ground_truth_data = []

        for i,sample in enumerate(out_samples):
            response = sample["response"]
            findings = sample["findings"]
            impression = sample["impression"]
            golden = f"Findings: {findings} Impression: {impression}."

            # 生成唯一的study_id
            study_id = f"study_{i+1}"
            
            # 添加预测数据
            predictions_data.append({
                'study_id': study_id,
                'report': response
            })

            # 添加真实标签数据
            ground_truth_data.append({
                'study_id': study_id,
                'report': golden
            })

        # 创建DataFrame
        predictions_df = pd.DataFrame(predictions_data)
        ground_truth_df = pd.DataFrame(ground_truth_data)

        # 确保输出目录存在
        os.makedirs(self.output_path, exist_ok=True)
        prediction_path = os.path.join(self.output_path,'predictions.csv')
        ground_truth_path = os.path.join(self.output_path,'ground_truth.csv')
        
        # 保存为CSV文件
        predictions_df.to_csv(prediction_path, index=False)
        ground_truth_df.to_csv(ground_truth_path, index=False)

        print(f"Predictions saved to {prediction_path}")
        print(f"Ground truth saved to {ground_truth_path}")

        return {"total metrics":"please use official evaluation script to generate metrics"}, out_samples