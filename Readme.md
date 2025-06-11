<p align="center">
    <img src="assets/logo.jpg" width="16%" height="25%">
    <img src="assets/title.png" width="55%" height="55%">
</p>

<h3 align="center">
MedEvalKit: A Unified Medical Evaluation Framework
</h3>

<font size=3><div align='center' > [[📖 arXiv Paper](https://arxiv.org/pdf/2506.07044)]</div></font>


<p align="center">
<a href="https://opensource.org/license/apache-2-0"><img src="https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg"></a>
<a href="https://github.com/alibaba-damo-academy"><img src="https://img.shields.io/badge/Institution-DAMO-red"></a>
<a><img src="https://img.shields.io/badge/PRs-Welcome-red"></a>
</p>

## Introduction
We focus on the evaluation framework for LMMs and LLMs in the medical field. We welcome everyone to provide some new models or benchmarks or better evaluation indicators.


## 🔥 News
* **[2025.6.12]**  The first version of MedEvalKit!


## Supported Benchmarks
Multi-Modal Medical benchmarks:
--MMMU-Medical-test
--MMMU-Medical-val
--PMC_VQA
--OmniMedVQA
--MedXpertQA-MM
--IU_XRAY
--CheXpert_Plus
--MIMIC_CXR
--VQA_RAD
--SLAKE
--PATH_VQA
--MedFrameQA

Text-only benchmarks:
--MedQA_USMLE
--MedMCQA
--PubMedQA
--Medbullets_op4
--Medbullets_op5
--MedXpertQA-Text
--SuperGPQA
--HealthBench
--CMB
--CMExam
--CMMLU
--MedQA_MCMLE

# Supported LMMs
hf only:
--BiMediX2
--BiomedGPT
--HealthGPT
--Janus
--Med_Flamingo
--MedDr
--MedGemma
--NVILA
--VILA_M3

both hf and vllm:
--HuatuoGPT-vision
--InternVL
--Llama_3.2-vision
--LLava
--LLava_Med
--Qwen2_5_VL
--Qwen2_VL



## 🛠️ Requirements and Installation
**Step 1**: Install the dependency
```bash
git clone https://github.com/DAMO-NLP-SG/MedEvalKit
cd MedEvalKit
pip install -r requirements.txt

pip install 'open_clip_torch[training]'
pip install flash-attn --no-build-isolation
```

For llava-like model(BiMediX2)
```bash
git clone https://github.com/LLaVA-VL/LLaVA-NeXT.git
cd LLaVA-NeXT
pip install -e .
```

**Step 2**: Prepare the dataset
HF datasets(You can use it directly by setting DATASETS_PATH to hf):
--VQA-RAD:flaviagiammarino/vqa-rad
--SuperGPQA:m-a-p/SuperGPQA
--PubMedQA:openlifescienceai/pubmedqa
--PATHVQA:flaviagiammarino/path-vqa
--MMMU: MMMU/MMMU
--MedQA-USMLE: GBaker/MedQA-USMLE-4-options
--MedQA-MCMLE: shuyuej/MedQA-MCMLE-Benchmark
--Medbullets_op4: tuenguyen/Medical-Eval-MedBullets_op4
--Medbullets_op5: LangAGI-Lab/medbullets_op5
--CMMMU: haonan-li/cmmlu
--CMExam: fzkuji/CMExam
--CMB: FreedomIntelligence/CMB
--MedFrameQA: SuhaoYu1020/MedFrameQA

Local datasets(you should download the datasets to the local directory):
--MedXpertQA: TsinghuaC3I/MedXpertQA
--SlAKE:BoKelvin/SLAKE
--PMCVQA:RadGenome/PMC-VQA 
--OmniMedVQA:foreverbeliever/OmniMedVQA
--Report Generation
  --MIMIC_CXR: https://physionet.org/content/mimic-cxr/2.1.0/ 
  --IU_Xray: https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university
  --CheXpert Plus: https://stanfordaimi.azurewebsites.net/datasets/5158c524-d3ab-4e02-96e9-6ee9efc110a1
--HealthBench: 
  --normal = "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/2025-05-07-06-14-12_oss_eval.jsonl"
  --hard = "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/hard_2025-05-08-21-00-10.jsonl"
  --consensus = "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/consensus_2025-05-09-20-00-46.jsonl"



## 🚀 Quick Start
**Step 1**: Set the eval.sh

Example
```bash
#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com
# MMMU-Medical-test,MMMU-Medical-val,PMC_VQA,MedQA_USMLE,MedMCQA,PubMedQA,OmniMedVQA,Medbullets_op4,Medbullets_op5,MedXpertQA-Text,MedXpertQA-MM,SuperGPQA,HealthBench,IU_XRAY,CheXpert_Plus,MIMIC_CXR,CMB,CMExam,CMMLU,MedQA_MCMLE,VQA_RAD,SLAKE,PATH_VQA
EVAL_DATASETS="MMMU-Medical-test,MMMU-Medical-val,PMC_VQA" # Evaluate the three benchmarks
DATASETS_PATH="hf" # set to hf if you do not download the benchmarks to the local else set it to your local path
OUTPUT_PATH="test" # All intermediate results and final evaluation results will be stored here
# TestModel,Qwen2-VL,Qwen2.5-VL,Qwen2.5,LLava_Med,Huatuo,InternVL,Llama-3.2,LLava,DeepSeek,Janus,HealthGPT,BiomedGPT,Vllm_Text
MODEL_NAME="Qwen2.5-VL"
MODEL_PATH="/mnt/workspace/workgroup_dev/longli/models/hub/Qwen2.5-VL-7B-Instruct"

#vllm setting
CUDA_VISIBLE_DEVICES="0,1"
TENSOR_PARALLEL_SIZE="2"
USE_VLLM="True"

#Eval setting
SEED=42
REASONING="False"
TEST_TIMES=1


# Eval LLM setting
MAX_NEW_TOKENS=8192
MAX_IMAGE_NUM=6
TEMPERATURE=0
TOP_P=0.0001
REPETITION_PENALTY=1

# LLM judge setting
USE_LLM_JUDGE="False"
# gpt-4o,gpt-4o-mini,gpt-4.1
GPT_MODEL="gpt-4.1"


# 运行 Python 脚本并传递参数
python eval.py \
    --eval_datasets "$EVAL_DATASETS" \
    --datasets_path "$DATASETS_PATH" \
    --output_path "$OUTPUT_PATH" \
    --model_name "$MODEL_NAME" \
    --model_path "$MODEL_PATH" \
    --seed $SEED \
    --cuda_visible_devices "$CUDA_VISIBLE_DEVICES" \
    --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
    --use_vllm "$USE_VLLM" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --max_image_num "$MAX_IMAGE_NUM" \
    --temperature "$TEMPERATURE"  \
    --top_p "$TOP_P" \
    --repetition_penalty "$REPETITION_PENALTY" \
    --reasoning "$REASONING" \
    --use_llm_judge "$USE_LLM_JUDGE" \
    --judge_gpt_model "$GPT_MODEL" \
    --test_times "$TEST_TIMES" 
```

**Step 2**: Run eval
```python
./eval.sh
```

## 📖 Evaluation Data
All evaluation results can be found in `OUTPUT_PATH` fold.

## :black_nib: Citation
If you find our work helpful for your research, please consider starring the repo and citing our work.   

```bibtex
@article{li2024chain,
  title={Chain of Ideas: Revolutionizing Research in Novel Idea Development with LLM Agents},
  author={Li, Long and Xu, Weiwen and Guo, Jiayan and Zhao, Ruochen and Li, Xinxuan and Yuan,
            Yuqian and Zhang, Boqiang and Jiang, Yuming and Xin, Yifei and Dang, Ronghao and 
            Rong, Yu and Zhao, Deli and Feng, Tian and Bing, Lidong},
  journal={arXiv preprint arXiv:2410.13185},
  year={2024},
  url={https://arxiv.org/abs/2410.13185}
}
```
