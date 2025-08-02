import os
import json
import math
from tqdm import tqdm
from .chunking import chunk_markdown
from .config import final_df, TOKEN_PER_QUESTION, EXCLUDED_FILES
from .prompts import return_prompt
from .parser import extract_pairs
from .llm_runner import init_llm, sanitize_filename

def get_questions(markdown_file, llm, question_type, fewshot_examples, ex_flag, output_file):
    id = int(markdown_file.split('/')[-1].split('.')[0].split('_')[0])
    markdown_chunks = chunk_markdown(markdown_file)
    qa_pairs, error, res = [], [], []

    title, state, program_type, sector_type = final_df[final_df['id'] == id][['name', 'state_name', 'program_category_name', 'sector_name']].to_numpy()[0]
    chunk_to_response = {}

    if ex_flag:
        with open(output_file, 'r', encoding='utf-8') as f:
            filedata = json.load(f)
            chunk_to_response = {doc['chunk']: doc['response'] for doc in filedata}

    for chunk in tqdm(markdown_chunks, desc='Processing chunks', leave=False):
        if chunk in chunk_to_response:
            response = chunk_to_response[chunk]
            qa_pairs.append({'chunk': chunk, 'response': response, 'question_type': question_type, 'document_id': id, 'llm': llm.model})
        else:
            tokens_in_chunk = len(chunk)
            num_questions = 5 if 'meta' in markdown_file else math.ceil(tokens_in_chunk / TOKEN_PER_QUESTION)
            prompt = return_prompt(title, state, program_type, sector_type, chunk, question_type, fewshot_examples, num_questions)
            response = llm.complete(prompt)
            response_text = response.text if hasattr(response, "text") else str(response)

            if response_text == 'NA': continue
            response = extract_pairs(response_text)
            if response_text: 
                res.append(response_text)
                qa_pairs.append({'chunk': chunk, 'response': response, 'question_type': question_type, 'document_id': id, 'llm': llm.model})
            else:
                error.append(response_text)

    return qa_pairs, error, res

def extract_qa_pairs(markdown_file, output_folder, model, question_type, fewshot_examples):
    fName = markdown_file.split('/')[-1].split('.')[0].strip()
    if fName in EXCLUDED_FILES:
        return None

    llm = init_llm(model)
    qtype_safe = question_type.replace('/', '-').replace(' ', '-')
    output_folder = f'{output_folder}/final/qa-gen/{fName}'
    os.makedirs(output_folder, exist_ok=True)

    model_safe = sanitize_filename(model)
    output_file = f'{output_folder}/{fName}_{model_safe}_{qtype_safe}_{fewshot_examples}.json'
    ex_flag = os.path.exists(output_file)

    qa_pairs, error, res = get_questions(markdown_file, llm, question_type, fewshot_examples, ex_flag, output_file)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(qa_pairs, f, indent=4, ensure_ascii=False)

    return qa_pairs, error, res
