import torch
import os
import json
import gc
import csv

from PIL import Image
from datasets import load_dataset,load_from_disk
from collections import defaultdict
from tqdm import tqdm


from mathruler.grader import extract_boxed_content
from ..utils import save_json,extract,judge_multi_choice,judger,get_compare_messages
from ..base_dataset import BaseDataset

from ..question_formats import get_close_ended_prompt,get_open_ended_prompt

class SLAKE(BaseDataset):
    def __init__(self,model,dataset_path,output_path):
        self.model = model
        self.output_path = output_path
        self.dataset_path = dataset_path
        self.samples = []
        self.chunk_idx = int(os.environ.get("chunk_idx",0))
        self.num_chunks = int(os.environ.get("num_chunks",1))

    
    def load_data(self):
        dataset_path = self.dataset_path
        dataset = []
        test_json_path = os.path.join(dataset_path,"test.json")
        with open(test_json_path,"r", encoding='utf-8') as f:
            datas = json.load(f)
        for data in datas:
            img_path = data["img_name"]
            question = data["question"]
            answer = data["answer"]
            answer_type = data["answer_type"]
            lang = data["q_lang"]

            is_reasoning = True if os.environ.get("REASONING","False") == "True" else False
            if answer_type == "OPEN":
                prompt = get_open_ended_prompt(question,is_reasoning,lang)
            else:
                prompt = get_close_ended_prompt(question,is_reasoning,lang)

            img_path = os.path.join(dataset_path,"imgs",img_path)
            image = Image.open(img_path)
            dataset.append({"image":image,"lang":lang,"answer_type":answer_type,"answer":answer,"question":question,"prompt":prompt})
        for idx,sample in tqdm(enumerate(dataset)):
            if idx % self.num_chunks == self.chunk_idx:
                sample = self.construct_messages(sample)
                self.samples.append(sample)
        return self.samples

    def construct_messages(self,sample):
        prompt = sample["prompt"]
        image = sample["image"]
        messages = {"prompt":prompt,"image":image}
        sample["messages"] = messages

        del sample["image"]
        return sample


    def cal_metrics(self,out_samples):
        messages_list = []

        total = {"close" : 0, "open":0,"en":0, "zh":0}
        right = {"close" : 0, "open":0,"en":0, "zh":0}

        open_id = []

        langs = []
        answer_types = []
        for out_sample in tqdm(out_samples):
            response = out_sample["response"]
            if extract_boxed_content(response)!= "None":
                response = extract_boxed_content(response)

            answer = out_sample["answer"]
            question = out_sample["question"]
            lang = out_sample["lang"]
            answer_type = out_sample["answer_type"]
            messages = get_compare_messages(question,response,answer,lang,answer_type)
            messages_list.append(messages)
            langs.append(lang)
            answer_types.append(answer_type)

        
        llm = judger
        results = llm.generate_outputs(messages_list)
        i = 0
        for result,lang,answer_type in zip(results,langs,answer_types):
            result = extract(result,"judge")
            result = True if result == "0" else False
            out_samples[i]["correct"] = result
            i += 1
            if answer_type == "OPEN":
                total["open"] += 1
                if result:
                    right["open"] += 1
            else:
                total["close"] += 1
                if result:
                    right["close"] += 1
            if lang == "en":
                total["en"] += 1
                if result:
                    right["en"] += 1
            else:
                total["zh"] += 1
                if result:
                    right["zh"] += 1

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
            },
            "en" : {
                "right" : right["en"],
                "total" : total["en"],
                "acc" : right["en"]/total["en"]
            },
            "zh" : {
                "right" : right["zh"],
                "total" : total["zh"],
                "acc" : right["zh"]/total["zh"]
            }
        }

        return metrics,out_samples


                