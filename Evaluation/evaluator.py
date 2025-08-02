import json
from .prompts import get_prompt
from .config import evaluation_model
from llama_index.llms.ollama import Ollama

def evaluate_qa_pairs(chunk, question, answer, context, conditions, metric_type, question_type,
                      llm, title, summary, program_type, sector_type, incentive_amount_data):
    try:
        prompt = get_prompt(chunk, question, answer, context, conditions,
                            metric_type, question_type, title, summary, program_type, sector_type, incentive_amount_data)
        response = llm.complete(prompt)
        return response.text
    except Exception as e:
        return False
