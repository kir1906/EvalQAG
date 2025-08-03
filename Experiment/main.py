import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import os
from sklearn.metrics import f1_score
from bert_score import score as bert_score
import re
import string
import collections  
from tqdm import tqdm
from collections import defaultdict

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace. 
    Copied from SQuAD 2.0 evaluation script."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s):
    """Split the normalized text into tokens."""
    if not s:
        return []
    return normalize_answer(s).split()

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = str(s)
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        return ''.join(ch for ch in text if ch not in set(string.punctuation))

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()

def compute_answer_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())

    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0

    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    return (2 * precision * recall) / (precision + recall)

def compute_conditions_f1(predicted_conditions, true_conditions):
    """Compute F1 of the predicted set of conditions."""
    if not true_conditions:
        return float(not predicted_conditions)
    if not predicted_conditions:
        return 0.0

    true_conditions = list(set(true_conditions))
    predicted_conditions = list(set(predicted_conditions))

    correct = sum([int(c in true_conditions) for c in predicted_conditions])
    precision = correct / len(predicted_conditions)
    recall = correct / len(true_conditions)

    if correct == 0.0:
        return 0.0
    else:
        return 2.0 / (1.0 / precision + 1.0 / recall)

def compute_em_f1(qtype, pred_answer, ref_answer):
    """
    Compute EM and F1 and conditional scores for one answer.

    Arguments:
    qtype: question type (e.g., 'yes_no_conditional')
    pred_answer: tuple (answer_text, conditions)
    ref_answer: tuple (answer_text, conditions)

    Returns:
    em, conditional_em, f1, conditional_f1
    """
    conditions_f1 = 0.0

    if 'cond' in qtype:
        conditions_f1 = compute_answer_f1(ref_answer[1],pred_answer[1])

    pred_answer_text = normalize_answer(pred_answer[0])
    ref_answer_text = normalize_answer(ref_answer[0])

    em = float(pred_answer_text == ref_answer_text)
    f1 = compute_answer_f1(ref_answer_text, pred_answer_text)

    conditional_em = em * conditions_f1
    conditional_f1 = f1 * conditions_f1

    return em, conditional_em, f1, conditional_f1

def prompt(question, context, qtype, isContextual=True):
    """
    Generate a prompt for the QA model based on the question type.

    Args:
        question (str): The question to answer.
        context (str): The supporting context.
        qtype (str): The question type (e.g., 'yes_no', 'factual', etc.)

    Returns:
        str: A full prompt string to send to the model.
    """
    instruction = "Provide a concise and well-formulated answer based only on the context."
    
    if "yes" in qtype.lower():
        instruction = (
            "Provide a clear Yes or No answer based strictly on the context. "
            "Avoid any elaboration. Return only 'Yes' or 'No'."
        )
    
    return f"""You are answering a "{qtype}" type question using the given context. Your goal is to produce an accurate, grammatically correct, and contextually grounded answer.

Instructions:
- {instruction}


Question: {question}
{f'Context: {context}' if isContextual else ''}

- Return JSON object with "answer" key and nothing else.

"""


def yes_no_w_cond_prompt(question, context, isContextual=True):
    """
    Generate a prompt specifically for 'Yes/No with conditions' questions.

    Args:
        question (str): The Yes/No question.
        context (str): The supporting context.

    Returns:
        str: A structured prompt for Yes/No with conditions questions.
    """
    return f"""You are answering a "Yes/No with conditions" type question based on the given context. Your task is to determine whether the answer is "Yes" or "No", and clearly list all conditions or assumptions that affect the answer.

Instructions:
- Begin your response with "Yes" or "No".
- Follow with a list of all conditions, scenarios, or clarifying factors relevant to the answer.
- Be precise and professional in your language.

Question: {question}
{f'Context: {context}' if isContextual else ''}


- Return JSON object with "answer" key and "conditions" key.

"""

def extract_json(text):
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        return {}

    json_str = match.group(0)

    try:
        data = json.loads(json_str)

        return data
    except Exception as e:
        return {}


def generate_result(tokenizer, model, data, isContextual=True):
    results = []
    errors = []
    
    def process_llm_request(d):
        input_prompt = (
            prompt(d['question'], d['context'], d['question_type'], isContextual) 
            if 'cond' not in d['question_type'] 
            else yes_no_w_cond_prompt(d['question'], d['context'], isContextual)
        )

        inputs = tokenizer(input_prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                top_p=0.9,
                temperature=0.8,
                pad_token_id=tokenizer.eos_token_id
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_text = generated_text.replace(input_prompt, "").strip()

        if '{' not in generated_text:
            generated_text = '{' + generated_text
        


        out = extract_json(generated_text)
        temp = d.copy()
        
        if not out or (d['question_type'] == 'Yes/No cond' and not 'conditions' in out):
            return None
        
        temp['result'] = out
        return temp


    for d in tqdm(data, desc="Processing data"):
        res = process_llm_request(d)
        
        if not res: errors.append(d)
        else:   results.append(res)
        
    
    while errors:
        d = errors.pop()
        res = process_llm_request(d)
        
        if not res: errors.append(d)
        else: results.append(res)
  
    return results

def compute_metrics(results):
    def parse_conditions(conds):
        if isinstance(conds, str):
            conds = conds.strip()
            return [] if conds.lower() == "none" or conds == "" else [c.strip() for c in conds.split(",")]
        return conds if isinstance(conds, list) else []

    new_data = []
    global_scores = {"em": 0, "f1": 0, "cond_em": 0, "cond_f1": 0, "count": 0}
    qtype_scores = defaultdict(lambda: {"em": 0, "f1": 0, "cond_em": 0, "cond_f1": 0, "count": 0})

    for obj in results:
        qtype = obj["question_type"]
        gt_answer = obj["answer"]
        gt_conditions = parse_conditions(obj.get("conditions", []))
        pred = obj.get("result", {})
        pred_answer = pred.get("answer", "")
        pred_conditions = parse_conditions(pred.get("conditions", []))

        em, cond_em, f1, cond_f1 = compute_em_f1(qtype, (pred_answer, str(pred_conditions)), (gt_answer, str(gt_conditions)))
        obj['f1'] = f1
        obj['conditional'] = cond_f1
        new_data.append(obj.copy())
        
        # Global aggregation
        global_scores["em"] += em
        global_scores["f1"] += f1
        global_scores["cond_em"] += cond_em
        global_scores["cond_f1"] += cond_f1
        global_scores["count"] += 1

        # Per-question-type aggregation
        qtype_scores[qtype]["em"] += em
        qtype_scores[qtype]["f1"] += f1
        qtype_scores[qtype]["cond_em"] += cond_em
        qtype_scores[qtype]["cond_f1"] += cond_f1
        qtype_scores[qtype]["count"] += 1
    
    global_scores = {
        'EM': global_scores['em'] / global_scores['count'],
        'F1': global_scores['f1'] / global_scores['count'],
        'cond_EM': global_scores['cond_em'] / global_scores['count'],
        'cond_F1': global_scores['cond_f1'] / global_scores['count'],
    }

    for qtype, scores in qtype_scores.items():
        count = scores["count"]
        global_scores[f'{qtype}_EM'] = scores['em'] / count
        global_scores[f'{qtype}_F1'] = scores['f1'] / count
        
        if 'cond' in qtype:
            global_scores[f'{qtype}_Cond_EM'] = scores['cond_em'] / count
            global_scores[f'{qtype}_Cond_F1'] = scores['cond_f1'] / count
    
    return results, global_scores

def save_json(data, path):
    """Utility function to save JSON with indentation"""
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)


def main(finetuned_model_path, data_path):
    
    # Load fine-tuned model and tokenizer
    tokenizer_finetuned = AutoTokenizer.from_pretrained(finetuned_model_path)
    model_finetuned = AutoModelForCausalLM.from_pretrained(
        finetuned_model_path, torch_dtype=torch.float16, device_map="auto"
    )

    # Load base model and tokenizer (Meta-LLaMA 3 8B Instruct)
    base_model_name = 'meta-llama/Meta-Llama-3-8B-Instruct'
    tokenizer_base = AutoTokenizer.from_pretrained(base_model_name)
    model_base = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.float16, device_map="auto"
    )

    # Load QA dataset
    with open(data_path) as f:
        data = json.load(f)
    
    data = data[:20]

    # Define all evaluation configurations
    eval_configs = [
        {
            "name": "finetuned_with_context",
            "tokenizer": tokenizer_finetuned,
            "model": model_finetuned,
            "is_contextual": True,
        },
        {
            "name": "finetuned_without_context",
            "tokenizer": tokenizer_finetuned,
            "model": model_finetuned,
            "is_contextual": False,
        },
        {
            "name": "base_with_context",
            "tokenizer": tokenizer_base,
            "model": model_base,
            "is_contextual": True,
        },
        {
            "name": "base_without_context",
            "tokenizer": tokenizer_base,
            "model": model_base,
            "is_contextual": False,
        },
    ]
    
    os.makedirs("./output", exist_ok=True)

    # Evaluate each configuration and save outputs
    for config in eval_configs:
        print(f"Running: {config['name']}")

        results = generate_result(
            tokenizer=config["tokenizer"],
            model=config["model"],
            data=data,
            isContextual=config["is_contextual"]
        )

        # Compute detailed evaluation and global scores
        eval_metrics, global_scores = compute_metrics(results)

        # Save outputs
        save_json(eval_metrics, f"./output/{config['name']}_eval_results.json")
        save_json(global_scores, f"./output/{config['name']}_global_scores.json")



if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python main.py <finetuned_model_path> <data_path>")
        sys.exit(1)

    finetuned_model_path = sys.argv[1]
    data_path = sys.argv[2]

    if not os.path.exists(finetuned_model_path):
        print(f"Error: The finetuned model path '{finetuned_model_path}' does not exist.")
        sys.exit(1)

    if not os.path.exists(data_path):
        print(f"Error: The data path '{data_path}' does not exist.")
        sys.exit(1)

    main(finetuned_model_path, data_path)
