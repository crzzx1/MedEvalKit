# /home/ubuntu/MedEvalKit/utils/question_formats.py
# This is the final, complete, and syntactically correct version.

# --- Highly Optimized Expert Prompts for PATH_VQA ---

def get_pathvqa_open_prompt(question, is_reasoning=True):
    """Generates a highly optimized, expert-level prompt for open-ended questions in PATH_VQA."""
    role_playing = "You are a board-certified pathologist interpreting a histology slide to answer a clinical question."
    few_shot_example = """
--- EXAMPLE START ---
Question: what is showing increased eosinophilia of cytoplasm, and swelling of occasional cells?
Microscopic Description: The image shows cardiac muscle cells. Some cells exhibit hyper-eosinophilic cytoplasm (pinker than normal) and cellular swelling. The nuclei appear pyknotic in some instances. Cross-striations are still visible.
Pathological Interpretation: These features - hypereosinophilia, cellular swelling, and pyknotic nuclei - are classic early signs of coagulation necrosis resulting from ischemia. Because cross-striations are preserved, it indicates the injury is in its early, potentially reversible, stages.
Final Answer: <answer>early (reversible) ischemic injury</answer>
--- EXAMPLE END ---
"""
    cot_instructions = """
Follow these steps to derive your answer:
1.  **Microscopic Description**: Based on the image, provide a concise description of the key histological features relevant to the question.
2.  **Pathological Interpretation**: Correlate the microscopic findings with your pathological knowledge to interpret the underlying condition or process.
3.  **Final Answer**: State your final, definitive answer based on your interpretation.

Please provide your final answer within <answer>...</answer> tags.
"""
    final_prompt = f"""{role_playing}

{few_shot_example}

Now, please solve the following task.
{cot_instructions}

Question: {question}
"""
    return final_prompt


def get_pathvqa_judgement_prompt(question, is_reasoning=True):
    """Generates a highly optimized, expert-level prompt for closed-ended (yes/no) questions in PATH_VQA."""
    role_playing = "You are a board-certified pathologist."
    cot_instructions = """
To answer the following yes/no question, first verify the presence or absence of the key features mentioned in the question, then provide a definitive 'yes' or 'no' conclusion.

Please provide your final answer within <answer>...</answer> tags.
"""
    final_prompt = f"""{role_playing}

{cot_instructions}

Question: {question}
Answer: <answer>"""
    return final_prompt


# --- General-Purpose Prompts for Other Datasets ---

def get_open_ended_prompt(question, is_reasoning=False, lang="en"):
    """Generic prompt for open-ended questions."""
    return f"Question: {question}\nAnswer:"


def get_judgement_prompt(question, is_reasoning=False, lang="en"):
    """Generic prompt for yes/no questions."""
    return f"Based on the image, answer the following question with 'yes' or 'no'.\nQuestion: {question}\nAnswer:"


def get_close_ended_prompt(question, is_reasoning=False, lang="en"):
    """
    Generic prompt for closed-ended questions.
    This is an alias for get_judgement_prompt to ensure compatibility with datasets like SLAKE.
    """
    return get_judgement_prompt(question, is_reasoning, lang)


def get_multiple_choice_prompt(question, choices, is_reasoning=True, lang="en"):
    """Prompt for multiple-choice datasets."""
    role_playing = "You are an expert physician taking a board exam."
    formatted_choices = "\n".join([f"{chr(ord('A') + i)}. {choice}" for i, choice in enumerate(choices)])

    base_prompt = f"""{role_playing}

The following is a multiple-choice question. Please choose the single best answer from the options provided.

Question: {question}

Options:
{formatted_choices}
"""

    if is_reasoning:
        cot_prompt = base_prompt + """
Analyze each option and provide a step-by-step reasoning for why it is correct or incorrect.
Conclude with the letter of the correct option. Your final answer should be a single letter in the format: <answer>X</answer>.

Reasoning:
Final Answer:
<answer>"""
        return cot_prompt
    else:
        simple_prompt = base_prompt + """
Your answer should be a single letter corresponding to the correct option.
Answer: <answer>"""
        return simple_prompt


def get_report_generation_prompt(is_reasoning=True):
    """
    Prompt for report generation tasks (e.g., chest X-ray reports for IU_XRAY).
    """
    role_playing = "You are an expert radiologist writing a diagnostic report for a chest X-ray."

    instructions = """Based on the provided chest X-ray image(s), please generate a structured radiological report. The report should include the following sections:

1.  **FINDINGS**: Detail your observations of the heart, lungs, pleura, and other visible structures.
2.  **IMPRESSION**: Provide a concise summary of the most critical findings and your overall diagnosis.

Example:
FINDINGS: The heart size is normal. The lungs are clear without evidence of consolidation, effusion, or pneumothorax.
IMPRESSION: No acute cardiopulmonary abnormalities.

Now, generate the report for the given image(s).
"""

    final_prompt = f"{role_playing}\n\n{instructions}"
    return final_prompt