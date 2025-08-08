#!/bin/bash
# 这个脚本会逐个将S3中的数据集同步到本地进行评测，以获得最佳I/O性能。

# set -e: 确保任何命令一旦失败，脚本将立即停止执行。
set -e
export HF_ENDPOINT=https://hf-mirror.com

# --- 1. 请根据您的实际情况修改以下配置 ---
S3_BUCKET="pi3-vlm"
S3_PARENT_PATH="data/medical/test_lingshu"
LOCAL_PARENT_PATH="/home/ubuntu/medeval_temp_data"
#    SLAKE
#    HealthBench
#    IU-Xray
# 数组中存放的是S3上的真实目录名
DATASET_ARRAY=(
    MIMIC_CXR
    MedXpertQA-Text
    MedXpertQA-MM
    OmniMedVQA
    PMC_VQA
)

# --- 2. 模型及评测参数配置 ---
OUTPUT_PATH="eval_results/{}"
MODEL_NAME="Qwen2.5-VL"
MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"
CUDA_VISIBLE_DEVICES="0"
TENSOR_PARALLEL_SIZE="1"
USE_VLLM="False"
SEED=42
REASONING="False"
TEST_TIMES=1
MAX_NEW_TOKENS=8192
MAX_IMAGE_NUM=6
TEMPERATURE=0
TOP_P=0.0001
REPETITION_PENALTY=1
USE_LLM_JUDGE="False"
GPT_MODEL="gpt-4.1-2025-04-14"
JUDGE_MODEL_TYPE="openai"
API_KEY=""
BASE_URL=""

# --- 3. 脚本主逻辑 ---

echo "✅ 启动本地评测流程..."
mkdir -p "$LOCAL_PARENT_PATH"
trap 'echo "正在清理临时目录..."; rm -rf "$LOCAL_PARENT_PATH"' EXIT

for dataset_name in "${DATASET_ARRAY[@]}"; do
    echo "========================================"
    echo "⚙️ 开始处理数据集: ${dataset_name}"
    echo "========================================"
    
    local_dataset_path="${LOCAL_PARENT_PATH}/${dataset_name}"

    echo "  - (1/3) 正在从 S3 同步数据到本地: ${local_dataset_path}"
    aws s3 sync "s3://${S3_BUCKET}/${S3_PARENT_PATH}/${dataset_name}/" "$local_dataset_path" --quiet

    # --- 核心修正点 ---
    # 默认情况下，python脚本使用和S3目录名一样的名称
    python_dataset_name="$dataset_name"
    # 但对于 IU-Xray，S3目录名(IU-Xray)和Python内部名(IU_XRAY)不同，需要特殊处理
    if [ "$dataset_name" == "IU-Xray" ]; then
        echo "  - 检测到 IU-Xray 数据集，使用 Python 内部名称 IU_XRAY 进行评估。"
        python_dataset_name="IU_XRAY"
    fi
    # --- 修正完毕 ---

    echo "  - (2/3) 正在对本地数据进行评测..."
    # 在调用 python eval.py 时使用修正后的名字
    python eval.py \
        --eval_datasets "$python_dataset_name" \
        --datasets_path "$LOCAL_PARENT_PATH" \
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
        --judge_model_type "$JUDGE_MODEL_TYPE" \
        --judge_model "$GPT_MODEL" \
        --api_key "$API_KEY" \
        --base_url "$BASE_URL" \
        --test_times "$TEST_TIMES"

    echo "  - (3/3) 正在清理本地临时数据: ${local_dataset_path}"
    rm -rf "$local_dataset_path"
    
    echo "✅ 数据集 ${dataset_name} 处理完成。"
done

echo "========================================"
echo "🎉 所有本地数据集均已评测完毕！"
echo "========================================"