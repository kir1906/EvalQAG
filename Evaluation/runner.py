import os
import json
import re
import pandas as pd
from tqdm import tqdm
from llama_index.llms.ollama import Ollama

from .config import metric_types
from .evaluator import evaluate_qa_pairs
from .utils import parse_json, safe_int

def main(files_folder, output_folder, start_index):
    final_df = pd.read_json('./data/final_df.json')

    if not os.path.exists(files_folder):
        print(f'Error: {files_folder} does not exist!')
        return
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    files = sorted(os.listdir(files_folder))[start_index:]
    result_files = []
    for folder in files:
        result_files.extend(os.listdir(os.path.join(files_folder, folder)))
    result_files.sort()

    res, errors = [], []
    for file in tqdm(result_files, desc='Files'):
        fName = file.rsplit('.', 1)[0]
        question_type = fName.split('_')[-2]
        gen_model = fName.split('_')[-3]
        document_name = '_'.join(fName.split('_')[:-3])

        output_folder_n = os.path.join(output_folder, "final_kri", "qa-eval", document_name)
        os.makedirs(output_folder_n, exist_ok=True)

        output_file_path = os.path.join(output_folder_n, f"{fName}.json")
        file_path = os.path.join(files_folder, document_name, file)

        existing_data = []
        completed_chunks_text = set()

        if os.path.exists(output_file_path):
            with open(output_file_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            for chunk_data in existing_data:
                if isinstance(chunk_data, dict) and 'chunk' in chunk_data:
                    completed_chunks_text.add(chunk_data['chunk'])

        with open(file_path, 'r', encoding='utf-8') as f:
            input_json = json.load(f)
        all_input_chunks = set(doc['chunk'] for doc in input_json)

        if completed_chunks_text.issuperset(all_input_chunks):
            continue

        doc_id = int(re.match(r"(\d+)_", fName).group(1))
        summary_row = final_df[final_df["id"] == doc_id]
        if summary_row.empty:
            print(f"No summary found for doc_id: {doc_id}")
            continue

        summary = summary_row['summary'].values[0]
        title, program_type, sector_type, incentive_amount_data = summary_row[['name', 'program_category_name', 'sector_name', 'incentive_amount_data']].to_numpy()[0]

        errors_size = len(errors)
        for i, document in tqdm(list(enumerate(input_json)), desc=f'chunks {fName}', leave=False):
            if document['chunk'] in completed_chunks_text:
                continue

            chunk = document['chunk']
            question_type = document['question_type']
            qa_pairs = document['response']
            llm = Ollama(model='qwen3:8b', request_timeout=100, verbose=False, json_mode=True)

            for obj in tqdm(qa_pairs, f'Question | {i}', leave=False):
                for metric_type in tqdm(metric_types, desc='Metric', leave=False):
                    tries = 0
                    obj[f'{metric_type}_score'] = -1
                    while tries < 2:
                        temp = evaluate_qa_pairs(chunk, obj['question'], obj['answer'], obj['context'], obj['conditions'],
                                                 metric_type, question_type, llm, title, summary, program_type, sector_type, incentive_amount_data)

                        if temp is False:
                            tries += 1
                            if tries == 2: errors.append(obj.copy())
                            continue

                        temp = parse_json(temp)

                        if isinstance(temp, str):
                            score = safe_int(temp)
                            if score == -1:
                                tries += 1
                                if tries == 2: errors.append(obj.copy())
                                continue
                        elif isinstance(temp, dict):
                            if 'score' not in temp:
                                tries += 1
                                if tries == 2: errors.append(obj.copy())
                                continue
                            try:
                                score = float(temp['score'] or 0)
                            except:
                                tries += 1
                                if tries == 2: errors.append(obj.copy())
                                continue

                        obj[f'{metric_type}_eval'] = temp
                        obj[f'{metric_type}_score'] = score
                        break

        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(input_json, f, indent=4, ensure_ascii=False)

        if len(errors) != errors_size:
            with open(os.path.join(output_folder, "error.json"), 'w', encoding='utf-8') as f:
                json.dump(errors, f, indent=4, ensure_ascii=False)

    with open(os.path.join(output_folder, "final.json"), 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=4, ensure_ascii=False)
