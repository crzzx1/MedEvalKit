# utils/MMLU/MMLU.py
# -----------------------------------------------------------
import os, json, gc, csv, re
from collections import defaultdict
from pathlib import Path

import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

# 引入 bert_score 用于第二阶段的语义相似度计算
from bert_score import score as bert_score_calc

from ..utils import save_json, extract
from ..base_dataset import BaseDataset
# 注意：我们不再使用旧的 get_multiple_choice_prompt，因为我们在下面直接构建Prompt
# from ..question_formats import get_multiple_choice_prompt

# 使用你原来的9个医学子领域定义
MEDICAL_SUBJECTS = [
    "anatomy",
    "clinical_knowledge",
    "college_biology",
    "college_medicine",
    "medical_genetics",
    "professional_medicine",
    "nutrition",
    "virology",
    "high_school_biology",
]


class MMLU(BaseDataset):
    """
    * subject = "medical" → 仅跑 9 门医学科目
    * subject = "all"     → 跑 57 全科
    """

    def __init__(self, model, dataset_path, output_path, subject="medical"):
        super().__init__()
        self.model = model
        self.output_path = Path(output_path)
        self.dataset_path = dataset_path or "cais/mmlu"
        self.subject = subject.lower()
        self.samples = []
        self.chunk_idx = int(os.getenv("chunk_idx", 0))
        self.num_chunks = int(os.getenv("num_chunks", 1))

    # ------------------------------------------------------------------
    # 读取 & 构建样本 - 使用你原来的9科目加载逻辑
    # ------------------------------------------------------------------
    def load_data(self):
        raw_examples = []

        if self.subject == "all":
            raw_examples = load_dataset(self.dataset_path, "all", split="test")
        elif self.subject == "medical":
            print(f"Loading 9 medical subjects: {MEDICAL_SUBJECTS}")
            for sub in MEDICAL_SUBJECTS:
                raw_examples += load_dataset(
                    self.dataset_path, sub, split="test"
                )
        else:
            raise ValueError(
                f"Unsupported subject '{self.subject}'. "
                "Choose from {'medical', 'all'}."
            )
        
        print(f"Total samples loaded: {len(raw_examples)}. Now constructing messages...")
        for idx, ex in enumerate(tqdm(raw_examples, desc="building prompts")):
            if idx % self.num_chunks != self.chunk_idx:
                continue
            self.samples.append(self._construct_messages(ex))

        return self.samples

    # ------------------------------------------------------------------
    # 构建Prompt - 使用优化后的新Prompt
    # ------------------------------------------------------------------
    def _construct_messages(self, ex):
        """把 HF 原始样本打包成 MedEvalKit 统一格式，并使用Lingshu论文的Prompt"""
        choices_list = ex["choices"]
        answer_idx = ex["answer"]
        question = ex["question"]
        alphas = ["A", "B", "C", "D"]

        formatted_choices = "\n".join([f"{alpha}. {choice}" for alpha, choice in zip(alphas, choices_list)])

        prompt = (
            f"Question: {question}\n"
            f"Options:\n"
            f"{formatted_choices}\n"
            f"Answer with the option's letter\n"
            f"from the given choices directly."
        )

        ex["messages"] = {"prompt": prompt}
        ex["answer"]   = alphas[answer_idx]
        # ex["choices"] 中已经是原始选项列表，可以直接给cal_metrics使用
        return ex

    # ------------------------------------------------------------------
    # 计算指标 - 使用优化后的两阶段评估策略
    # ------------------------------------------------------------------
    def cal_metrics(self, out_samples):
        # --- 辅助函数定义 ---
        def parse_answer_option(text):
            """第一阶段：使用正则表达式从文本中提取单个选项字母 (A, B, C, D)"""
            if not isinstance(text, str):
                return None
            match = re.search(r'\b([A-D])\b', text)
            if match:
                return match.group(1)
            return None

        # --- 主逻辑开始 ---
        total = 0
        right = 0
        
        print("Starting two-stage evaluation...")
        for i, sample in tqdm(enumerate(out_samples), total=len(out_samples)):
            response = sample["response"]
            choices_text_list = sample["choices"]
            correct_answer_letter = sample["answer"]
            correct = False
            
            # --- 第一阶段：规则匹配 ---
            predicted_letter = parse_answer_option(response)
            
            if predicted_letter is not None:
                if predicted_letter == correct_answer_letter:
                    correct = True
            
            # --- 第二阶段：语义相似度匹配 (如果第一阶段失败) ---
            else:
                try:
                    cands = [response] * len(choices_text_list)
                    refs = choices_text_list
                    
                    P, R, F1 = bert_score_calc(cands, refs, lang="en", model_type="bert-base-uncased", verbose=False)
                    best_option_idx = torch.argmax(F1).item()
                    
                    semantic_predicted_letter = ["A", "B", "C", "D"][best_option_idx]
                    
                    if semantic_predicted_letter == correct_answer_letter:
                        correct = True
                except Exception as e:
                    print(f"Warning: bert-score failed for a sample, treating as incorrect. Error: {e}")
                    correct = False

            if correct:
                right += 1
            
            out_samples[i]["is_correct"] = correct
            total += 1

        acc = right / total if total > 0 else 0
        metrics = {"total metrics": {"total": total, "right": right, "acc": acc}}
        print(f"Final Accuracy (Two-Stage Evaluation on 9 subjects): {acc:.4f} ({right}/{total})")
        return metrics, out_samples