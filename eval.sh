#!/bin/bash
#export HF_ENDPOINT=https://hf-mirror.com
##MMMU-Medical-test,MMMU-Medical-val,MedQA_USMLE,MedMCQA,PubMedQA,Medbullets_op4,Medbullets_op5,SuperGPQA,CMB,CMExam,PATH_VQA,MedFrameQA,MedXpertQA-Text,SLAKE,PMC_VQA,MMLU
# MMMU-Medical-test,MMMU-Medical-val,PMC_VQA,MedQA_USMLE,MedMCQA,PubMedQA,OmniMedVQA,Medbullets_op4,Medbullets_op5,MedXpertQA-Text,MedXpertQA-MM,SuperGPQA,HealthBench,IU_XRAY,CheXpert_Plus,MIMIC_CXR,CMB,CMExam,CMMLU,MedQA_MCMLE,VQA_RAD,SLAKE,PATH_VQA,MedFrameQA
EVAL_DATASETS="MedXpertQA-MM,OmniMedVQA"  #MIMIC_CXR,HealthBench(不用）),OmniMedVQA(太大了)  IU_XRAY ,MedXpertQA-MM(太大了)
#SLAKE,HealthBench,IU_XRAY,MIMIC_CXR,MedXpertQA-Text,MedXpertQA-MM,OmniMedVQA,PMC_VQA,   #CheXpert_Plus,HealthBench,  IU_XRAY,MedXpertQA-MM
DATASETS_PATH="/home/ubuntu/linshu-test" #/home/ubuntu/linshu-test
OUTPUT_PATH="eval_results/{}"
# TestModel,Qwen2-VL,Qwen2.5-VL,BiMediX2,LLava_Med,Huatuo,InternVL,Llama-3.2,LLava,Janus,HealthGPT,BiomedGPT,Vllm_Text,MedGemma,Med_Flamingo,MedDr
MODEL_NAME="Qwen2.5-VL"
MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"

#vllm setting
CUDA_VISIBLE_DEVICES="0"
TENSOR_PARALLEL_SIZE="1"
USE_VLLM="True" #True

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
GPT_MODEL="gemini-2.5-flash" # 已按要求修改为最新的 2.5 Flash 模型
JUDGE_MODEL_TYPE="gemini"         # 设置为 'gemini' 来调用 Google API
API_KEY="AIzaSyDR-6RTpKwdxnCfjEpdRP9UUn12zdHieM8" # 您的 API 密钥
BASE_URL=""                         # 使用官方API端点时通常留空


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
    --max-model-len 16032 \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --max_image_num "$MAX_IMAGE_NUM" \
    --temperature "$TEMPERATURE"  \
    --top_p "$TOP_P" \
    --repetition_penalty "$REPETITION_PENALTY" \
    --reasoning "$REASONING" \
    --use_llm_judge "$USE_LLM_JUDGE" \
    --judge_model_type "$JUDGE_MODEL_TYPE" \
    --judge_model "$GPT_MODEL" \
    --api_key "$API_KEY" \
    --base_url "$BASE_URL" \
    --test_times "$TEST_TIMES" \
