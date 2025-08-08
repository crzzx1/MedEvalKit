# -----------------------------------------------------------
# utils/__init__.py 里的 prepare_benchmark  (替换版)
# -----------------------------------------------------------
from utils import (
    Medbullets_op4, Medbullets_op5, MedXpertQA, MMMU, OmniMedVQA,
    PATH_VQA, PMC_VQA, SLAKE, SuperGPQA, VQA_RAD, HealthBench,
    PubMedQA, MedMCQA, MedQA_USMLE, MMLU, CMB, CMExam, MedQA_MCMLE,
    CMMLU, IU_XRAY, CheXpert_Plus, MIMIC_CXR, MedFrameQA
)

# ------------------------------------------------------------------
# 支持的数据集清单
# ------------------------------------------------------------------
supported_dataset = [
    # --- 单选题 / 多选题 -----------------------------------------
    "MMLU", "MMLU-medical", "MMLU-all",
    "MMMU-Medical-test", "MMMU-Medical-val",
    "MedQA_USMLE", "MedMCQA", "PubMedQA", "MedQA_MCMLE",
    "CMB", "CMExam", "CMMLU",

    # --- 医学问答 -------------------------------------------------
    "Medbullets_op4", "Medbullets_op5",
    "MedXpertQA-Text", "MedXpertQA-MM",
    "SuperGPQA",

    # --- 多模态医学 VQA -------------------------------------------
    "PATH_VQA", "PMC_VQA", "VQA_RAD", "SLAKE", "OmniMedVQA",

    # --- 影像/报告 ------------------------------------------------
    "IU_XRAY", "CheXpert_Plus", "MIMIC_CXR",

    # --- 其它 -----------------------------------------------------
    "HealthBench", "MedFrameQA",
]

# ------------------------------------------------------------------
#  根据名字实例化对应数据集
# ------------------------------------------------------------------
def prepare_benchmark(model, eval_dataset, eval_dataset_path, eval_output_path):
    """
    返回一个数据集实例；若名称无效则返回 None 并打印提示
    """
    # ===== 1. MMLU 系列 =====
    #  允许三种写法：
    #   - MMLU           → subject="medical"   (默认 9 门医学科目)
    #   - MMLU-medical   → subject="medical"
    #   - MMLU-all       → subject="all"       (57 全科)
    if eval_dataset.startswith("MMLU"):
        if eval_dataset in ["MMLU", "MMLU-medical"]:
            subject = "medical"
        elif eval_dataset == "MMLU-all":
            subject = "all"
        else:          # 写错了子串
            print(f"unknown MMLU variant '{eval_dataset}', "
                  f"choose from {{MMLU, MMLU-medical, MMLU-all}}")
            return None

        # 替换路径中出现的 MMLU*（若有）
        if eval_dataset_path:
            eval_dataset_path = eval_dataset_path.replace(eval_dataset.split("-")[0], "MMLU")

        return MMLU(model, eval_dataset_path, eval_output_path, subject)

    # ===== 2. 继续保持原有分支 =====
    if eval_dataset in ["MMMU-Medical-test", "MMMU-Medical-val"]:
        if eval_dataset_path:
            eval_dataset_path = eval_dataset_path.replace(eval_dataset, "MMMU")
        _, subset, split = eval_dataset.split("-")
        return MMMU(model, eval_dataset_path, eval_output_path, split, subset)

    if eval_dataset == "PATH_VQA":
        return PATH_VQA(model, eval_dataset_path, eval_output_path)

    if eval_dataset == "PMC_VQA":
        return PMC_VQA(model, eval_dataset_path, eval_output_path)

    if eval_dataset == "VQA_RAD":
        return VQA_RAD(model, eval_dataset_path, eval_output_path)

    if eval_dataset == "SLAKE":
        return SLAKE(model, eval_dataset_path, eval_output_path)

    if eval_dataset == "OmniMedVQA":
        return OmniMedVQA(model, eval_dataset_path, eval_output_path)

    if eval_dataset == "Medbullets_op4":
        return Medbullets_op4(model, eval_dataset_path, eval_output_path)

    if eval_dataset == "Medbullets_op5":
        return Medbullets_op5(model, eval_dataset_path, eval_output_path)

    if eval_dataset in ["MedXpertQA-Text", "MedXpertQA-MM"]:
        if eval_dataset_path:
            eval_dataset_path = eval_dataset_path.replace(eval_dataset, "MedXpertQA")
        _, split = eval_dataset.split("-")
        return MedXpertQA(model, eval_dataset_path, eval_output_path, split)

    if eval_dataset == "SuperGPQA":
        return SuperGPQA(model, eval_dataset_path, eval_output_path)

    if eval_dataset == "HealthBench":
        return HealthBench(model, eval_dataset_path, eval_output_path)

    if eval_dataset == "PubMedQA":
        return PubMedQA(model, eval_dataset_path, eval_output_path)

    if eval_dataset == "MedMCQA":
        return MedMCQA(model, eval_dataset_path, eval_output_path)

    if eval_dataset == "MedQA_USMLE":
        return MedQA_USMLE(model, eval_dataset_path, eval_output_path)

    if eval_dataset == "CMB":
        return CMB(model, eval_dataset_path, eval_output_path)

    if eval_dataset == "CMExam":
        return CMExam(model, eval_dataset_path, eval_output_path)

    if eval_dataset == "MedQA_MCMLE":
        return MedQA_MCMLE(model, eval_dataset_path, eval_output_path)

    if eval_dataset == "CMMLU":
        return CMMLU(model, eval_dataset_path, eval_output_path)

    if eval_dataset == "IU_XRAY":
        return IU_XRAY(model, eval_dataset_path, eval_output_path)

    if eval_dataset == "CheXpert_Plus":
        return CheXpert_Plus(model, eval_dataset_path, eval_output_path)

    if eval_dataset == "MIMIC_CXR":
        return MIMIC_CXR(model, eval_dataset_path, eval_output_path)

    if eval_dataset == "MedFrameQA":
        return MedFrameQA(model, eval_dataset_path, eval_output_path)

    # ===== 3. 名字不在列表 =====
    print(f"unknown eval dataset '{eval_dataset}', "
          f"supported list: {supported_dataset}")
    return None
