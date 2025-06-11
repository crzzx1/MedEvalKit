import torch
import os
import json
import gc
import csv

from PIL import Image
from datasets import load_dataset
from collections import defaultdict
from tqdm import tqdm
from mathruler.grader import extract_boxed_content

from ..utils import save_json,extract,judge_multi_choice,judger,get_compare_messages
from ..base_dataset import BaseDataset
from ..question_formats import get_judgement_prompt,get_open_ended_prompt

class VQA_RAD(BaseDataset):
    def __init__(self,model,dataset_path,output_path):
        self.model = model
        self.output_path = output_path
        self.dataset_path = dataset_path if dataset_path else "flaviagiammarino/vqa-rad"
        self.samples = []
        self.chunk_idx = int(os.environ.get("chunk_idx",0))
        self.num_chunks = int(os.environ.get("num_chunks",1))

    
    def load_data(self):
        dataset_path = self.dataset_path
        dataset = load_dataset(dataset_path,split = "test")
            
        for idx,sample in tqdm(enumerate(dataset)):
            if idx % self.num_chunks == self.chunk_idx:
                sample = self.construct_messages(sample)
                self.samples.append(sample)
        return self.samples

    def construct_messages(self,sample):
        question = sample["question"]
        image = sample["image"]
        answer = sample["answer"]
        
        is_reasoning = True if os.environ.get("REASONING","False") == "True" else False
        answer = answer.lower()
        if answer in ["yes","no"]:
            prompt = get_judgement_prompt(question,is_reasoning)
        else:
            prompt = get_open_ended_prompt(question,is_reasoning)


        messages = {"prompt":prompt,"image":image}
        sample["messages"] = messages
        del sample["image"]
        return sample


    def cal_metrics(self,out_samples):
        messages_list = []

        total = {"close" : 0, "open":0}
        right = {"close" : 0, "open":0}

        open_id = []
        for i,out_sample in tqdm(enumerate(out_samples)):
            response = out_sample["response"]
            response = extract(response,"answer")
            if extract_boxed_content(response)!= "None":
                response = extract_boxed_content(response)

            answer = out_sample["answer"]
            question = out_sample["question"]
            answer = answer.lower().strip()
            response = response.lower().strip()
            response = response.split(".")[0]
            if answer in ["yes","no"]:
                total["close"] += 1
                correct = False
                if answer == response:
                    right["close"] += 1
                    correct = True
                out_samples[i]["correct"] = correct
            else:
                total["open"] += 1
                messages = get_compare_messages(question,response,answer)
                messages_list.append(messages)
                open_id.append(i)
                
        llm = judger
        results = llm.generate_outputs(messages_list)
        for id,result in zip(open_id,results):
            result = extract(result,"judge")
            result = True if result == "0" else False
            out_samples[id]["correct"] = result
            if result:
                right["open"] += 1

        
        metrics = {
            "total metrics" : {
                "right" : right["open"] + right["close"],
                "total" : total["open"] + total["close"],
                "acc" : (right["open"] + right["close"]) / (total["open"] + total["close"])
            },
            "open-ended" : {
                "right" : right["open"],
                "total" : total["open"],
                "acc" : right["open"]/total["open"]
            },
            "close-ended" : {
                "right" : right["close"],
                "total" : total["close"],
                "acc" : right["close"]/total["close"]
            }
        }
        return metrics,out_samples


                