<h3 align="center">
  🩺 MedEvalKit: A Unified Medical Evaluation Framework
</h3>

<p align="center">
  <a href="https://arxiv.org/abs/2506.07044" target="_blank">📖 arXiv Paper</a> •
  <a href="https://huggingface.co/collections/lingshu-medical-mllm/lingshu-mllms-6847974ca5b5df750f017dad" target="_blank">🤗 Lingshu Models</a> •
  <a href="https://alibaba-damo-academy.github.io/lingshu/" target="_blank">🌐 Lingshu Project Page</a>
</p>

<p align="center">
  <a href="https://opensource.org/license/apache-2-0">
    <img src="https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg" alt="License">
  </a>
  <a href="https://github.com/alibaba-damo-academy">
    <img src="https://img.shields.io/badge/Institution-DAMO-red" alt="Institution">
  </a>
  <a>
    <img src="https://img.shields.io/badge/PRs-Welcome-red" alt="PRs Welcome">
  </a>
</p>

---

## 📌 Introduction
A comprehensive evaluation framework for **Large Medical Models (LMMs/LLMs)** in the healthcare domain.  
We welcome contributions of new models, benchmarks, or enhanced evaluation metrics!

---

## Eval Results
| Models                | MMMU-Med | VQA-RAD | SLAKE | PathVQA | PMC-VQA | OmniMedVQA | MedXpertQA | Avg. |
|-----------------------|:-------:|:-------:|:-----:|:-------:|:-------:|:----------:|:----------:|:----:|
| **Proprietary Models**|    –    |    –    |   –   |    –    |    –    |      –     |      –     |  –   |
| GPT-4.1               |  75.2   |  65.0   | 72.2  |  55.5   |  55.2   |    75.5    |    45.2    | 63.4 |
| Claude Sonnet 4       |  74.6   |  67.6   | 70.6  |  54.2   |  54.4   |    65.5    |    43.3    | 61.5 |
| Gemini-2.5-Flash      |  76.9   |  68.5   | 75.8  |  55.4   |  55.4   |    71.0    |    52.8    | 65.1 |
| **Open-source Models (<10 B)**| – | – | – | – | – | – | – | – |
| BiomedGPT♡            |  24.9   |  16.6   | 13.6  |  11.3   |  27.6   |    27.9    |      –     |  –   |
| Med-R1-2B◇            |  34.8   |  39.0   | 54.5  |  15.3   |  47.4   |      –     |    21.1    |  –   |
| MedVLM-R1-2B          |  35.2   |  48.6   | 56.0  |  32.5   |  47.6   |    77.7    |    20.4    | 45.4 |
| MedGemma-4B-IT        |  43.7   |  72.5   | 76.4  |  48.8   |  49.9   |    69.8    |    22.3    | 54.8 |
| LLaVA-Med-7B          |  29.3   |  53.7   | 48.0  |  38.8   |  30.5   |    44.3    |    20.3    | 37.8 |
| HuatuoGPT-V-7B        |  47.3   |  67.0   | 67.8  |  48.0   |  53.3   |    74.2    |    21.6    | 54.2 |
| BioMediX2-8B          |  39.8   |  49.2   | 57.7  |  37.0   |  43.5   |    63.3    |    21.8    | 44.6 |
| Qwen2.5VL-7B          |  50.6   |  64.5   | 67.2  |  44.1   |  51.9   |    63.6    |    22.3    | 52.0 |
| InternVL2.5-8B        |  53.5   |  59.4   | 69.0  |  42.1   |  51.3   |    81.3    |    21.7    | 54.0 |
| InternVL3-8B          |  59.2   |  65.4   | 72.8  |  48.6   |  53.8   |    79.1    |    22.4    | 57.3 |
| Lingshu-7B            |  54.0   |  67.9   | 83.1  |  61.9   |  56.3   |    82.9    |    26.7    | 61.8 |
| **Open-source Models (>10 B)**| – | – | – | – | – | – | – | – |
| HealthGPT-14B         |  49.6   |  65.0   | 66.1  |  56.7   |  56.4   |    75.2    |    24.7    | 56.2 |
| HuatuoGPT-V-34B       |  51.8   |  61.4   | 69.5  |  44.4   |  56.6   |    74.0    |    22.1    | 54.3 |
| MedDr-40B♡            |  49.3   |  65.2   | 66.4  |  53.5   |  13.9   |    64.3    |      –     |  –   |
| InternVL3-14B         |  63.1   |  66.3   | 72.8  |  48.0   |  54.1   |    78.9    |    23.1    | 58.0 |
| Qwen2.5V-32B          |  59.6   |  71.8   | 71.2  |  41.9   |  54.5   |    68.2    |    25.2    | 56.1 |
| InternVL2.5-38B       |  61.6   |  61.4   | 70.3  |  46.9   |  57.2   |    79.9    |    24.4    | 57.4 |
| InternVL3-38B         |  65.2   |  65.4   | 72.7  |  51.0   |  56.6   |    79.8    |    25.2    | 59.4 |
| Lingshu-32B           |  62.3   |  76.5   | 89.2  |  65.9   |  57.9   |    83.4    |    30.9    | 66.6 |




## 🔥 Latest News
* **2025-06-12** - Initial release of MedEvalKit v1.0!

---

## 🧪 Supported Benchmarks

| Multimodal Medical Benchmarks | Text-Only Medical Benchmarks |
|-----------------------|----------------------|
| MMMU-Medical-test     | MedQA-USMLE          |
| MMMU-Medical-val      | MedMCQA              |
| PMC_VQA               | PubMedQA             |
| OmniMedVQA            | Medbullets-op4       |
| IU XRAY               | Medbullets-op5       |
| MedXpertQA-Multimodal | MedXpertQA-Text      |
| CheXpert Plus         | SuperGPQA            |
| MIMIC-CXR             | HealthBench          |
| VQA-RAD               | CMB                  |
| SLAKE                 | CMExam               |
| PATH-VQA              | CMMLU                |
| MedFrameQA            | MedQA-MCMLE          |

---

## 🤖 Supported Models
### HuggingFace Exclusive
<div style="column-count: 2;">

* BiMediX2
* BiomedGPT
* HealthGPT
* Janus
* Med_Flamingo
* MedDr
* MedGemma
* NVILA
* VILA_M3

</div>

### HF + vLLM Compatible
<div style="column-count: 2;">

* HuatuoGPT-vision
* InternVL
* Llama_3.2-vision
* LLava
* LLava_Med
* Qwen2_5_VL
* Qwen2_VL

</div>

---

## 🛠️ Installation
```bash
# Clone repository
git clone https://github.com/DAMO-NLP-SG/MedEvalKit
cd MedEvalKit

# Install dependencies
pip install -r requirements.txt
pip install 'open_clip_torch[training]'
pip install flash-attn --no-build-isolation

# For LLaVA-like models
git clone https://github.com/LLaVA-VL/LLaVA-NeXT.git
cd LLaVA-NeXT && pip install -e .
```

---

## 📂 Dataset Preparation
### HuggingFace Datasets (Direct Access)
```python
# Set DATASETS_PATH='hf'
VQA-RAD: flaviagiammarino/vqa-rad
SuperGPQA: m-a-p/SuperGPQA
PubMedQA: openlifescienceai/pubmedqa
PATHVQA: flaviagiammarino/path-vqa
MMMU: MMMU/MMMU
MedQA-USMLE: GBaker/MedQA-USMLE-4-options
MedQA-MCMLE: shuyuej/MedQA-MCMLE-Benchmark
Medbullets_op4: tuenguyen/Medical-Eval-MedBullets_op4
Medbullets_op5: LangAGI-Lab/medbullets_op5
CMMMU: haonan-li/cmmlu
CMExam: fzkuji/CMExam
CMB: FreedomIntelligence/CMB
MedFrameQA: SuhaoYu1020/MedFrameQA
```

### Local Datasets (Manual Download Required)
| Dataset          | Source |
|------------------|--------|
| MedXpertQA       | [TsinghuaC3I](https://huggingface.co/datasets/TsinghuaC3I/MedXpertQA) |
| SLAKE            | [BoKelvin](https://huggingface.co/datasets/BoKelvin/SLAKE) |
| PMCVQA           | [RadGenome](https://huggingface.co/datasets/RadGenome/PMC-VQA) |
| OmniMedVQA       | [foreverbeliever](https://huggingface.co/datasets/foreverbeliever/OmniMedVQA) |
| MIMIC_CXR        | [MIMIC_CXR](https://physionet.org/content/mimic-cxr/2.1.0/) |
| IU_Xray          | [IU_Xray](https://openi.nlm.nih.gov/faq?download=true) |
| CheXpert Plus    | [CheXpert Plus](https://aimi.stanford.edu/datasets/chexpert-plus) |
| HealthBench       | [Normal](https://openaipublic.blob.core.windows.net/simple-evals/healthbench/2025-05-07-06-14-12_oss_eval.jsonl),[Hard](https://openaipublic.blob.core.windows.net/simple-evals/healthbench/hard_2025-05-08-21-00-10.jsonl),[Consensus](https://openaipublic.blob.core.windows.net/simple-evals/healthbench/consensus_2025-05-09-20-00-46.jsonl) |

---

## 🚀 Quick Start
### 1. Configure `eval.sh`
```bash
#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com
# MMMU-Medical-test,MMMU-Medical-val,PMC_VQA,MedQA_USMLE,MedMCQA,PubMedQA,OmniMedVQA,Medbullets_op4,Medbullets_op5,MedXpertQA-Text,MedXpertQA-MM,SuperGPQA,HealthBench,IU_XRAY,CheXpert_Plus,MIMIC_CXR,CMB,CMExam,CMMLU,MedQA_MCMLE,VQA_RAD,SLAKE,PATH_VQA,MedFrameQA
EVAL_DATASETS="Medbullets_op4" 
DATASETS_PATH="hf"
OUTPUT_PATH="eval_results/{}"
# TestModel,Qwen2-VL,Qwen2.5-VL,BiMediX2,LLava_Med,Huatuo,InternVL,Llama-3.2,LLava,Janus,HealthGPT,BiomedGPT,Vllm_Text,MedGemma,Med_Flamingo,MedDr
MODEL_NAME="Qwen2.5-VL"
MODEL_PATH="Qwen2.5-VL-7B-Instruct"

#vllm setting
CUDA_VISIBLE_DEVICES="0"
TENSOR_PARALLEL_SIZE="1"
USE_VLLM="False"

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
USE_LLM_JUDGE="True"
# gpt api model name
GPT_MODEL="gpt-4.1-2025-04-14"
OPENAI_API_KEY=""


# pass hyperparameters and run python sccript
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
    --openai_api_key "$OPENAI_API_KEY" \
    --test_times "$TEST_TIMES" 
```

### 2. Run Evaluation
```bash
chmod +x eval.sh  # Add execute permission
./eval.sh
```

---

## 📜 Citation
```bibtex
@article{xu2025lingshu,
  title={Lingshu: A Generalist Foundation Model for Unified Multimodal Medical Understanding and Reasoning},
  author={Xu, Weiwen and Chan, Hou Pong and Li, Long and Aljunied, Mahani and Yuan, Ruifeng and Wang, Jianyu and Xiao, Chenghao and Chen, Guizhen and Liu, Chaoqun and Li, Zhaodonghui and others},
  journal={arXiv preprint arXiv:2506.07044},
  year={2025}
}
```

<div align="center">
  <sub>Built with ❤️ by the DAMO Academy Medical AI Team</sub>
</div>
